"""LC Classification schedules (parsed 2024 PDFs from ~/git/lcc_pdfs_2024) as a
queryable index.

Indexes every caption node of every schedule tree (numbered or not) with its parent,
so a class number can be resolved to: the printed entry (or the narrowest range that
contains it), the full ancestor chain WITH the notes attached at each level (G 058:
instructions "may appear dozens of pages before the class number"), the tables
referenced on the way, and the sibling/child entries.  Tables (P-PZ40, H77a, E1 ...)
are indexed by code with their rows.

First use builds .cache/lcc.sqlite (~1-2 min); set SHELFLISTER_LCC_DIR to override the
JSON directory.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

LCC_DIR = Path(os.environ.get("SHELFLISTER_LCC_DIR", Path.home() / "git" / "lcc_pdfs_2024" / "json"))
DB_PATH = Path(os.environ.get("SHELFLISTER_LCC_DB", Path(__file__).resolve().parent.parent / "build" / "lcc.sqlite"))

_NUM = re.compile(r"^(\d+)(?:\.(\d+))?((?:\.?[A-Z]\d*)*)$")


def number_key(section: str, num: str | None, *, end: bool = False) -> str | None:
    """Sortable key for a schedule number within a section. 'A-Z' style cutter ends
    ('.Z') are widened so that any Cutter starting with that letter is contained."""
    if not num:
        return None
    n = num.strip().lstrip(".")
    m = _NUM.match(n)
    if not m:
        return None
    integer, dec, cutters = m.groups()
    dec = (dec or "").ljust(8, "0")[:8] if not end else (dec or ("9" * 8 if False else "")).ljust(8, "0")[:8]
    # decimals compare as strings after left-align padding with zeros: 5 -> 50000000, 55 -> 55000000
    parts = re.findall(r"([A-Z])(\d*)", cutters or "")
    ck = ""
    for letter, digits in parts:
        if not digits and end:
            ck += letter + "~"          # .Z with no digits = everything starting with Z
        else:
            ck += letter + digits
    if end and not parts:
        ck = "~"                          # whole-number end of range: include all cutters under it
    return f"{section:<3}{int(integer):05d}.{dec}{ck}"


def call_number_key(class_number: str) -> tuple[str, str] | None:
    """(section, key) for a call number's class part, e.g. 'TX749.5.B43' -> ('TX', ...)."""
    m = re.match(r"^\s*([A-Z]{1,3})\s*(\d.*)$", class_number.strip())
    if not m:
        return None
    section, rest = m.group(1), re.sub(r"\s+.*$", "", m.group(2))   # drop book number / date
    k = number_key(section, rest)
    return (section, k) if k else None


@dataclass
class Node:
    id: int
    schedule: str
    section: str
    kind: str
    number: str | None
    caption: str
    notes: list[str]
    table: str | None
    status: str | None
    page: int | None
    parent: int | None
    depth: int

    def label(self) -> str:
        num = f"{self.section}{self.number}" if self.number else ""
        s = f"{num:<18} {self.caption}" if num else f"{'':<18} {self.caption}"
        if self.table:
            s += f"  [Table {self.table}]"
        if self.status and self.status != "valid":
            s += f"  ({self.status})"
        for n in self.notes:
            s += f"\n{'':<19}  note: {n}"
        return s


@dataclass
class Lookup:
    class_number: str
    exact: Node | None
    containing: list[Node]        # narrowest last
    ancestors: list[Node]         # root first, for the narrowest node
    children: list[Node]
    siblings: list[Node]
    tables: dict[str, "Table"]

    @property
    def narrowest(self) -> Node | None:
        return self.exact or (self.containing[-1] if self.containing else None)

    def instructions(self) -> list[str]:
        """Every note on the path (schedule instructions that override the CSM)."""
        out = []
        for n in self.ancestors + ([self.narrowest] if self.narrowest else []):
            out.extend(f"{n.section}{n.number or ''} {n.caption}: {x}" for x in n.notes)
        return out

    def text(self, max_children: int = 40) -> str:
        lines = [f"Schedule lookup for {self.class_number}:"]
        if not self.narrowest:
            lines.append("  (no entry found in the 2024 schedules)")
            return "\n".join(lines)
        lines.append("  Hierarchy (root first; notes are schedule instructions and take precedence over the CSM):")
        for i, n in enumerate(self.ancestors):
            lines.append("  " + "  " * i + n.label().replace("\n", "\n" + "  " * (i + 1)))
        d = len(self.ancestors)
        tag = "EXACT" if self.exact else "CONTAINING RANGE"
        lines.append("  " + "  " * d + f"{tag}: " + self.narrowest.label().replace("\n", "\n" + "  " * (d + 1)))
        if self.children:
            lines.append(f"  Entries printed under it ({len(self.children)}):")
            for c in self.children[:max_children]:
                lines.append("    " + c.label().replace("\n", "\n    "))
            if len(self.children) > max_children:
                lines.append(f"    ... {len(self.children) - max_children} more")
        elif self.siblings:
            lines.append(f"  Neighbouring entries ({len(self.siblings)}):")
            for c in self.siblings[:max_children]:
                lines.append("    " + c.label().replace("\n", "\n    "))
        for code, t in self.tables.items():
            lines.append(f"  Table {code}: {t.title}")
            lines.append(t.text(indent="    "))
        return "\n".join(lines)


@dataclass
class Table:
    code: str
    schedule: str
    title: str
    rows: list[dict] = field(default_factory=list)   # {number, caption, notes, level}

    def text(self, indent: str = "", max_rows: int = 80, find: str | None = None) -> str:
        out = []
        rows = self.rows
        if find:
            words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", find)]
            rows = [r for r in self.rows if any(w in (r["caption"] + " " + " ".join(r.get("notes") or [])).lower() for w in words)]
            out.append(f"{indent}({len(rows)} of {len(self.rows)} rows match {find!r})")
        if len(rows) > max_rows:
            out.append(f"{indent}({len(rows)} rows; first {max_rows} shown -- pass find=<words> to filter)")
        for r in rows[:max_rows]:
            out.append(f"{indent}{'  ' * r['level']}{(r['number'] or ''):<16} {r['caption']}")
            for n in r.get("notes") or []:
                out.append(f"{indent}{'  ' * r['level']}{'':<16}  note: {n}")
        return "\n".join(out)


class Schedules:
    def __init__(self, db_path: Path = DB_PATH, lcc_dir: Path = LCC_DIR):
        self.db_path, self.lcc_dir = db_path, lcc_dir
        if not db_path.exists():
            self.build()
        # read-only + immutable: no journal/shm files, so it works on a read-only filesystem (Lambda image)
        # check_same_thread=False: the shared instance is reused across Lambda invocations, whose worker threads
        # differ; the DB is immutable and only the shelflist task reads it, so this is safe.
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row

    # -- build ----------------------------------------------------------
    def build(self):
        if not self.lcc_dir.exists():
            raise FileNotFoundError(f"LCC JSON directory not found: {self.lcc_dir}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_suffix(".building")
        if tmp.exists():
            tmp.unlink()
        db = sqlite3.connect(tmp)
        db.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, schedule TEXT, section TEXT, kind TEXT, number TEXT,
                caption TEXT, notes TEXT, tbl TEXT, status TEXT, page INTEGER, parent INTEGER, depth INTEGER,
                start_key TEXT, end_key TEXT, ord INTEGER);
            CREATE TABLE tables (code TEXT PRIMARY KEY, schedule TEXT, title TEXT, rows TEXT);
            CREATE VIRTUAL TABLE caption_fts USING fts5(node_id UNINDEXED, section UNINDEXED, number UNINDEXED,
                caption, path, notes, tokenize='porter unicode61');
            CREATE TABLE index_terms (id INTEGER PRIMARY KEY, schedule TEXT, term TEXT, refs TEXT);
            CREATE VIRTUAL TABLE index_fts USING fts5(term_id UNINDEXED, term, tokenize='porter unicode61');
        """)
        nid = 0
        ordn = 0
        rows: list[tuple] = []
        fts: list[tuple] = []

        def walk(node: dict, schedule: str, section: str, kind: str, parent: int | None, depth: int, path: list[str]):
            nonlocal nid, ordn
            nid += 1
            ordn += 1
            me = nid
            num = node.get("number")
            sk = number_key(section, node.get("number_start") or num) if num else None
            ek = number_key(section, node.get("number_end") or node.get("number_start") or num, end=True) if num else None
            cap = node.get("caption", "")
            notes = node.get("notes") or []
            rows.append((me, schedule, section, kind, num, cap, json.dumps(notes),
                         node.get("table"), node.get("status"), node.get("page"), parent, depth, sk, ek, ordn))
            if num and kind == "subclass":
                fts.append((me, section, f"{section}{num}", cap, " -- ".join(path), " ".join(notes)))
            for ch in node.get("children", []):
                walk(ch, schedule, section, kind, me, depth + 1, path + [cap])

        for f in sorted(self.lcc_dir.glob("*.json")):
            if f.name in ("G_Cutter.json", "qa_report.json"):
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            schedule = d.get("schedule", f.stem)
            for sec in d.get("subclasses", []):
                code = sec.get("code") or ""
                for e in sec.get("entries", []):
                    walk(e, schedule, code, "subclass", None, 0, [])
            # index terms: "Term -- Subterm" with their class refs
            def idx_walk(n, prefix):
                term = " -- ".join(prefix + [n.get("term", "")]).strip()
                refs = n.get("refs") or []
                if term and refs:
                    cur = db.execute("INSERT INTO index_terms (schedule, term, refs) VALUES (?,?,?)", (schedule, term, json.dumps(refs)))
                    db.execute("INSERT INTO index_fts (term_id, term) VALUES (?,?)", (cur.lastrowid, term))
                for ch in n.get("children", []):
                    idx_walk(ch, prefix + [n.get("term", "")])
            for e in d.get("index", []):
                idx_walk(e, [])
            for t in d.get("tables", []):
                flat = []

                def flat_walk(n, level):
                    flat.append({"number": n.get("number"), "caption": n.get("caption", ""), "notes": n.get("notes") or [], "level": level})
                    for ch in n.get("children", []):
                        flat_walk(ch, level + 1)
                for e in t.get("entries", []):
                    flat_walk(e, 0)
                db.execute("INSERT OR REPLACE INTO tables VALUES (?,?,?,?)", (t.get("code"), schedule, t.get("title", ""), json.dumps(flat)))
            if len(rows) > 50000:
                db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                db.executemany("INSERT INTO caption_fts VALUES (?,?,?,?,?,?)", fts)
                rows.clear(); fts.clear()
        db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.executemany("INSERT INTO caption_fts VALUES (?,?,?,?,?,?)", fts)
        db.executescript("""
            CREATE INDEX ix_keys ON nodes(section, start_key, end_key);
            CREATE INDEX ix_parent ON nodes(parent);
            CREATE INDEX ix_num ON nodes(section, number);
        """)
        db.commit()
        db.close()
        tmp.rename(self.db_path)

    # -- queries --------------------------------------------------------
    def _node(self, r: sqlite3.Row) -> Node:
        return Node(r["id"], r["schedule"], r["section"], r["kind"], r["number"], r["caption"], json.loads(r["notes"]),
                    r["tbl"], r["status"], r["page"], r["parent"], r["depth"])

    def get(self, node_id: int) -> Node:
        return self._node(self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone())

    def ancestors(self, node: Node) -> list[Node]:
        out = []
        while node.parent is not None:
            node = self.get(node.parent)
            out.append(node)
        return list(reversed(out))

    def children(self, node: Node) -> list[Node]:
        return [self._node(r) for r in self.db.execute("SELECT * FROM nodes WHERE parent=? ORDER BY ord", (node.id,))]

    def table(self, code: str) -> Table | None:
        code = code.strip()
        r = self.db.execute("SELECT * FROM tables WHERE code=?", (code,)).fetchone()
        if not r:
            # tolerate 'Table P-PZ40', 'P-PZ40 modified'
            c = re.sub(r"^(?:Table\s+)?", "", code, flags=re.I).split()[0]
            r = self.db.execute("SELECT * FROM tables WHERE code=?", (c,)).fetchone()
        if not r:
            return None
        return Table(r["code"], r["schedule"], r["title"], json.loads(r["rows"]))

    def lookup(self, class_number: str) -> Lookup:
        sk = call_number_key(class_number)
        empty = Lookup(class_number, None, [], [], [], [], {})
        if not sk:
            return empty
        section, key = sk
        # contained in [start,end]; for a bare number (no Cutter) also the A-Z Cutter range
        # printed on that number, whose start key is the number followed by a letter
        bare = "." in key and key.split(".")[1][8:] == ""
        rows = self.db.execute(
            "SELECT * FROM nodes WHERE section=? AND start_key IS NOT NULL AND end_key>=? "
            "AND (start_key<=? OR (? AND start_key LIKE ?)) ORDER BY ord", (section, key, key, bare, key + "%")).fetchall()
        if not rows:
            return empty
        nodes = [self._node(r) for r in rows]
        # narrowest = the one with the tightest [start,end]; exact = start==end and equals the query
        exact = None
        for r, n in zip(rows, nodes):
            if r["start_key"] == key and (r["end_key"] == key or r["end_key"] == key + "~"):
                exact = n
        # narrowest last: deeper in the tree = narrower
        order = sorted(nodes, key=lambda n: n.depth)
        containing = [n for n in order if n is not exact]
        narrow = exact or (containing[-1] if containing else None)
        anc = self.ancestors(narrow) if narrow else []
        kids = self.children(narrow) if narrow else []
        sibs = self.children(anc[-1]) if (not kids and anc) else []
        tables: dict[str, Table] = {}
        for n in anc + ([narrow] if narrow else []):
            codes = []
            if n.table:
                codes.append(n.table)
            for note in n.notes:
                codes.extend(re.findall(r"\bTable\s+([A-Z][A-Za-z0-9\-]*)", note))
            codes.extend(re.findall(r"\(Table\s+([A-Z][A-Za-z0-9\-]*)", n.caption))
            for c in codes:
                t = self.table(c)
                if t and t.code not in tables:
                    tables[t.code] = t
        return Lookup(class_number, exact, containing, anc, kids, sibs, tables)


    # -- search ---------------------------------------------------------
    @staticmethod
    def _fts_query(q: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", q)
        return " OR ".join(f'"{w}"' for w in words if len(w) > 1) or '""'

    def search(self, query: str, limit: int = 20, section: str | None = None) -> list[dict]:
        """Full-text search over schedule captions (with ancestor path and notes) and the
        schedules' own index terms. Returns [{class_number, caption, path, notes, via}] ranked."""
        q = self._fts_query(query)
        out, seen = [], set()
        sql = ("SELECT node_id, number, caption, path, notes, bm25(caption_fts, 0, 0, 0, 10.0, 3.0, 1.0) AS r FROM caption_fts "
               "WHERE caption_fts MATCH ? " + ("AND section=? " if section else "") + "ORDER BY r LIMIT ?")
        args = (q, section, limit * 2) if section else (q, limit * 2)
        for r in self.db.execute(sql, args):
            if r["number"] in seen:
                continue
            seen.add(r["number"])
            out.append({"class_number": r["number"], "caption": r["caption"], "path": r["path"],
                        "notes": r["notes"], "via": "caption", "rank": r["r"]})
        for r in self.db.execute("SELECT t.term, t.refs, bm25(index_fts) AS r FROM index_fts f JOIN index_terms t ON t.id=f.term_id "
                                 "WHERE index_fts MATCH ? ORDER BY r LIMIT ?", (q, limit)):
            for ref in json.loads(r["refs"]):
                if section and not ref.startswith(section):
                    continue
                key = ref.rstrip("+")
                if key in seen:
                    continue
                seen.add(key)
                L = self.lookup(key)
                n = L.narrowest
                out.append({"class_number": ref, "caption": n.caption if n else "", "path": " -- ".join(a.caption for a in L.ancestors),
                            "notes": " ".join(n.notes) if n else "", "via": f"index: {r['term']}", "rank": r["r"]})
        out.sort(key=lambda x: x["rank"])
        return out[:limit]

    def search_text(self, query: str, limit: int = 20, section: str | None = None) -> str:
        hits = self.search(query, limit, section)
        if not hits:
            return "(no schedule captions or index terms match)"
        lines = [f"Schedule search for {query!r}:"]
        for h in hits:
            lines.append(f"  {h['class_number']:<16} {h['caption']}")
            if h["path"]:
                lines.append(f"      under: {h['path']}")
            if h["notes"]:
                lines.append(f"      notes: {h['notes'][:200]}")
            if h["via"] != "caption":
                lines.append(f"      ({h['via']})")
        return "\n".join(lines)


def expand_table_row(base_cutter: str, row_number: str) -> str:
    """Apply a '.x' table row to a base Cutter: ('O72', '.xA61-.xZ458') -> 'O72A61-O72Z458'."""
    return row_number.replace(".x", base_cutter).replace("-.", "-").replace(".", "", 1) if row_number else ""


_BIO_TABLES = {"E1", "N6", "N7", "H1", "H2"}   # tables shaped like the CSM biography table


def schedule_tags(lookup: "Lookup") -> set[str]:
    """Routing tags that follow mechanically from the schedule entry."""
    tags: set[str] = set()
    n = lookup.narrowest
    if not n:
        return tags
    captions = " ".join(a.caption for a in lookup.ancestors) + " " + n.caption
    notes = " ".join(lookup.instructions())
    if lookup.tables:
        tags.add("class:table_instruction")
    if re.search(r"\bBy date\b", n.caption) or re.search(r"\bby date\b", " ".join(n.notes), re.I):
        tags.add("class:by_date")
    if any(t in _BIO_TABLES for t in lookup.tables) or re.search(r"\bBiography\b.*\bIndividual\b|\bIndividual biography\b", captions):
        tags.add("class:biography_number")
    if re.search(r"\bTable P-PZ\d+", notes + " " + captions):
        tags.add("form:literary_author")
    if re.search(r"\.A1\b|\.A[1-5]-|Periodicals\. Societies\. Serials", " ".join(c.caption + " " + (c.number or "") for c in lookup.children)):
        tags.add("class:reserved_a_range")
    if re.search(r"\bregion or country\b|\bBy country\b|\bBy region\b", captions + notes, re.I):
        tags.add("class:has_geographic_cutter")
    if re.search(r"\bLaw\b", captions) and n.section.startswith("K"):
        tags.add("form:law")
    return tags


_SCHEDULES: "Schedules | None" = None


def get_schedules() -> "Schedules | None":
    """Shared instance; None when the LCC JSON is not available on this machine."""
    global _SCHEDULES
    if _SCHEDULES is None:
        try:
            _SCHEDULES = Schedules()
        except FileNotFoundError:
            return None
    return _SCHEDULES
