"""Semantic search over the LC classification schedules, all on AWS.

Every numbered caption (with its ancestor path) and every numbered table row is embedded once with
Amazon Titan Text Embeddings V2 (512 dims, normalized) through a Bedrock batch inference job, and the
vectors live in an S3 Vectors index. At query time the query is embedded with one Titan call and the
index is searched by cosine similarity, optionally filtered to a section (subclass letters) or a table.
`Schedules.search` merges these hits with its full-text hits, so a query can find "Islam and economics"
without sharing a word with the caption.

Build (one-off, ~$1):   uv run python -m shelflister.semantic export | submit | status | load | query "..."
Env: AWS creds for us-east-1; SHELFLISTER_VECTOR_BUCKET (shelflister-lcc-vectors), SHELFLISTER_VECTOR_INDEX
(captions), SHELFLISTER_SEMANTIC=0 to disable at query time.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REGION = os.environ.get("SHELFLISTER_AWS_REGION", "us-east-1")
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMS = 512
VECTOR_BUCKET = os.environ.get("SHELFLISTER_VECTOR_BUCKET", "shelflister-lcc-vectors")
VECTOR_INDEX = os.environ.get("SHELFLISTER_VECTOR_INDEX", "captions")
BATCH_BUCKET = os.environ.get("SHELFLISTER_BATCH_BUCKET", "innvo-embedded-catalog-local")
BATCH_PREFIX = "batch_embeddings/lcc_schedules"
BATCH_ROLE = os.environ.get("SHELFLISTER_BATCH_ROLE", "arn:aws:iam::323998827103:role/innov-bedrock-batch-role")
PART_SIZE = 50_000
WORK = Path(__file__).resolve().parent.parent / ".cache" / "vectors"
NON_FILTERABLE = ["caption", "path", "number", "node_id", "row"]


def _session():
    import boto3
    return boto3.Session(region_name=REGION)


# ------------------------------------------------------------------ texts
_GENERIC = re.compile(r"^(general( works)?( \(table [^)]*\))?|general|special topics(, a-z)?|other( special)?( topics)?(, a-z)?|"
                      r"by (region or country|country|period|language|subject)(, a-z)?|a-z|periodicals(\. societies\. serials)?|"
                      r"collections|collected works|individual( authors| works)?.*|biography|history|congresses|dictionaries.*|"
                      r"study and teaching.*|early works.*|\d{4}-|through \d{4}|\d{2}th century|\d{4}-\d{4}|[a-z]\b)$", re.I)


def is_generic(caption: str) -> bool:
    return bool(_GENERIC.match(caption.strip().rstrip(".")))


def caption_text(caption: str, path: str | None, extra: str | None = None, terms: list[str] | None = None,
                 notes: str | None = None) -> str:
    """What gets embedded. A generic leaf ("General works", "Special topics, A-Z", a period) says nothing on
    its own, so the nearest topical ancestor leads; then the caption, the ancestors nearest first, the
    schedule's own index terms that point here, and the notes ("Including ...", "Class here ...")."""
    anc = [p.strip() for p in re.split(r"\s+--\s+|\s+>\s+", path or "") if p.strip()]
    topical = next((a for a in reversed(anc) if not is_generic(a)), None) if is_generic(caption) else None
    parts = []
    if topical:
        parts.append(f"{topical}: {caption.strip()}")
    else:
        parts.append(caption.strip())
    if anc:
        parts.append(" > ".join(reversed(anc)))
    if terms:
        parts.append("index: " + "; ".join(t for t in terms[:6]))
    if notes:
        parts.append("note: " + notes.strip()[:200])
    if extra:
        parts.append(extra)
    return " | ".join(p for p in parts if p)[:1800]


def _index_terms(sch) -> dict[str, list[str]]:
    """class number -> index terms that reference it (the schedules' printed indexes)."""
    out: dict[str, list[str]] = {}
    for r in sch.db.execute("SELECT term, refs FROM index_terms"):
        term = re.sub(r"\s+--\s+in\s+--\s+", " in ", r["term"]).strip()
        for ref in json.loads(r["refs"]):
            out.setdefault(ref.rstrip("+"), []).append(term)
    return out


def iter_records(sch):
    """(record_id, text, metadata) for every numbered caption and numbered table row."""
    terms = _index_terms(sch)
    for r in sch.db.execute("SELECT f.node_id, f.number, f.caption, f.path, f.notes, f.section, n.schedule FROM caption_fts f "
                            "JOIN nodes n ON n.id = f.node_id"):
        yield (f"n{r['node_id']}", caption_text(r["caption"], r["path"], terms=terms.get(r["number"]), notes=r["notes"]),
               {"kind": "node", "section": r["section"] or "", "schedule": r["schedule"] or "", "table": "",
                "number": r["number"] or "", "caption": r["caption"][:500], "path": (r["path"] or "")[:1500], "node_id": str(r["node_id"]), "row": ""})
    for t in sch.db.execute("SELECT code, schedule, title, rows FROM tables"):
        rows = json.loads(t["rows"])
        stack: list[str] = []
        for i, row in enumerate(rows):
            lvl = row.get("level") or 0
            stack = stack[:lvl]
            path = " -- ".join(stack)
            stack = stack + [row["caption"]]
            if not row.get("number"):
                continue
            yield (f"t{t['code']}|{i}", caption_text(row["caption"], path, f"Table {t['code']} {t['title'] or ''}".strip(),
                                                   notes=" ".join(row.get("notes") or [])),
                   {"kind": "table", "section": "", "schedule": t["schedule"] or "", "table": t["code"],
                    "number": row["number"], "caption": row["caption"][:500], "path": path[:1500], "node_id": "", "row": str(i)})


# ------------------------------------------------------------------ build steps
def export(sch, work: Path = WORK) -> list[Path]:
    """Write batch-inference input parts (<= PART_SIZE records) and a metadata sidecar."""
    work.mkdir(parents=True, exist_ok=True)
    parts, n, f, meta = [], 0, None, (work / "metadata.jsonl").open("w")
    for rid, text, md in iter_records(sch):
        if n % PART_SIZE == 0:
            if f:
                f.close()
            p = work / f"part_{len(parts):04d}.jsonl"
            parts.append(p)
            f = p.open("w")
        f.write(json.dumps({"recordId": rid, "modelInput": {"inputText": text, "dimensions": DIMS, "normalize": True}}) + "\n")
        meta.write(json.dumps({"id": rid, **md}, ensure_ascii=False) + "\n")
        n += 1
    if f:
        f.close()
    meta.close()
    print(f"exported {n} records in {len(parts)} parts -> {work}")
    return parts


def submit(parts: list[Path]) -> list[str]:
    """Upload the parts and start one Bedrock batch job per part. Returns the job ARNs (also saved)."""
    s = _session()
    s3, br = s.client("s3"), s.client("bedrock")
    arns = []
    for p in parts:
        key = f"{BATCH_PREFIX}/input/{p.name}"
        s3.upload_file(str(p), BATCH_BUCKET, key)
        name = f"lcc-schedules-p{p.stem.split('_')[-1]}-{int(time.time())}"      # job names: letters, digits, - . + only
        for attempt in range(20):
            try:
                r = br.create_model_invocation_job(
                    jobName=name, modelId=MODEL_ID, roleArn=BATCH_ROLE,
                    inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{BATCH_BUCKET}/{key}"}},
                    outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{BATCH_BUCKET}/{BATCH_PREFIX}/output/{p.stem}/"}})
                arns.append(r["jobArn"])
                (WORK / "jobs.json").write_text(json.dumps(arns, indent=1))
                print(f"submitted {name}: {r['jobArn']}")
                break
            except Exception as e:  # noqa: BLE001
                if "ServiceQuotaExceeded" in str(e) or "Too many" in str(e) or "ThrottlingException" in str(e):
                    print(f"  quota/throttle on {name}, waiting 60s ({e})"); time.sleep(60)
                else:
                    raise
    (WORK / "jobs.json").write_text(json.dumps(arns, indent=1))
    return arns


def status(arns: list[str] | None = None) -> dict[str, str]:
    br = _session().client("bedrock")
    arns = arns or json.loads((WORK / "jobs.json").read_text())
    out = {}
    for a in arns:
        j = br.get_model_invocation_job(jobIdentifier=a)
        out[a] = j["status"] + (f" ({j.get('message')})" if j.get("message") else "")
    return out


def ensure_index():
    sv = _session().client("s3vectors")
    try:
        sv.get_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    except sv.exceptions.NotFoundException:
        sv.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        print("created vector bucket", VECTOR_BUCKET)
    try:
        sv.get_index(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX)
    except sv.exceptions.NotFoundException:
        sv.create_index(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX, dataType="float32", dimension=DIMS,
                        distanceMetric="cosine", metadataConfiguration={"nonFilterableMetadataKeys": NON_FILTERABLE})
        print("created index", VECTOR_INDEX)
    return sv


def load(work: Path = WORK) -> int:
    """Read the batch outputs from S3 and put the vectors (with metadata) into the index."""
    s = _session()
    s3, sv = s.client("s3"), ensure_index()
    meta = {}
    for line in (work / "metadata.jsonl").open():
        d = json.loads(line)
        meta[d.pop("id")] = d
    done_f = work / "loaded_parts.json"
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
    total = 0
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=BATCH_BUCKET, Prefix=f"{BATCH_PREFIX}/output/")
    keys = [o["Key"] for page in pages for o in page.get("Contents", []) if o["Key"].endswith(".jsonl.out")]
    for key in sorted(keys):
        if key in done:
            continue
        body = s3.get_object(Bucket=BATCH_BUCKET, Key=key)["Body"].iter_lines()
        batch, n, bad = [], 0, 0
        for line in body:
            d = json.loads(line)
            emb = (d.get("modelOutput") or {}).get("embedding")
            rid = d.get("recordId")
            if not emb or rid not in meta:
                bad += 1
                continue
            batch.append({"key": rid, "data": {"float32": emb}, "metadata": meta[rid]})
            if len(batch) == 500:
                sv.put_vectors(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX, vectors=batch); n += len(batch); batch = []
        if batch:
            sv.put_vectors(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX, vectors=batch); n += len(batch)
        done.add(key); done_f.write_text(json.dumps(sorted(done)))
        total += n
        print(f"loaded {n} vectors from {key} ({bad} without embedding)")
    return total


# ------------------------------------------------------------------ query
class SemanticSearch:
    """Query the index. Silently unavailable (returns []) when AWS is not reachable or disabled."""

    def __init__(self):
        self.enabled = os.environ.get("SHELFLISTER_SEMANTIC", "1") != "0"
        self._s = None

    def _clients(self):
        if self._s is None:
            s = _session()
            self._s = (s.client("bedrock-runtime"), s.client("s3vectors"))
        return self._s

    def embed(self, text: str) -> list[float]:
        rt, _ = self._clients()
        r = rt.invoke_model(modelId=MODEL_ID, body=json.dumps({"inputText": text[:2000], "dimensions": DIMS, "normalize": True}))
        return json.loads(r["body"].read())["embedding"]

    def search(self, query: str, k: int = 10, *, section: str | None = None, table: str | None = None,
               kind: str | None = None) -> list[dict]:
        if not self.enabled:
            return []
        try:
            _, sv = self._clients()
            flt: dict = {}
            if section:
                flt["section"] = {"$eq": section.upper()}
            if table:
                flt["table"] = {"$eq": table}
            elif kind:
                flt["kind"] = {"$eq": kind}
            elif not section:
                flt["kind"] = {"$eq": "node"}
            r = sv.query_vectors(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX, topK=k,
                                 queryVector={"float32": self.embed(query)}, returnMetadata=True, returnDistance=True,
                                 **({"filter": flt} if flt else {}))
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_warned", False):
                print(f"(semantic search unavailable: {type(e).__name__}: {str(e)[:120]})", file=sys.stderr); self._warned = True
            return []
        out = []
        for v in r.get("vectors", []):
            m = v.get("metadata") or {}
            cn = m.get("number", "")           # a node's number already carries its section letters (caption_fts.number)
            out.append({"class_number": cn, "caption": m.get("caption", ""), "path": m.get("path", ""),   # root first, as the FTS path
                        "notes": "", "via": "semantic" + (f": Table {m['table']}" if m.get("table") else ""),
                        "score": 1 - float(v.get("distance", 1)), "kind": m.get("kind"), "table": m.get("table", ""), "row": m.get("row", "")})
        return out


_SEARCH: SemanticSearch | None = None


def get_semantic() -> SemanticSearch:
    global _SEARCH
    if _SEARCH is None:
        _SEARCH = SemanticSearch()
    return _SEARCH


if __name__ == "__main__":
    from .schedule import get_schedules
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "export":
        export(get_schedules())
    elif cmd == "submit":
        submit(sorted(WORK.glob("part_*.jsonl")))
    elif cmd == "status":
        for a, st in status().items():
            print(st, a.rsplit("/", 1)[-1])
    elif cmd == "load":
        print("total loaded", load())
    elif cmd == "query":
        for h in get_semantic().search(" ".join(sys.argv[2:]), 10):
            print(f"{h['score']:.3f} {h['class_number']:<16} {h['caption'][:50]:<50} {h['path'][:70]}")
