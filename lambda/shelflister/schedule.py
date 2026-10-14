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
import threading
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
    via: str | None = None        # "Table KDZ-KH2 (Brazil)" when the entry was expanded from a jurisdiction table

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
        if self.via:
            tag += f" (expanded from {self.via})"
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
        self._local = threading.local()

    @property
    def db(self) -> sqlite3.Connection:
        """One read-only connection per thread (the eval runners query the schedules from worker threads)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # read-only + immutable: no journal/shm files, so it works on Lambda's read-only image filesystem
            conn = self._local.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
        return conn

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
        # for a bare number, the specific Cutters printed under it (G156.5.E26 ...) matched by prefix are its
        # entries, not ranges containing it; only the A-Z range itself (start ...A, end ...Z~) contains it
        extra_kids = []
        if bare:
            keep = []
            for r, n in zip(rows, nodes):
                if r["start_key"] > key and not (r["start_key"] == key + "A" and r["end_key"] >= key + "Z"):
                    extra_kids.append(n)
                else:
                    keep.append(n)
            nodes = keep
        # narrowest last: deeper in the tree = narrower
        order = sorted(nodes, key=lambda n: n.depth)
        containing = [n for n in order if n is not exact]
        narrow = exact or (containing[-1] if containing else None)
        anc = self.ancestors(narrow) if narrow else []
        kids = (self.children(narrow) if narrow else []) or extra_kids
        sibs = self.children(anc[-1]) if (not kids and anc) else []
        if narrow is not None and exact is None and re.search(r"\(Table\s+K", narrow.caption or ""):
            jur = next((j for j in self.jurisdictions() if j["node"].id == narrow.id), None)
            if jur:
                L = self._lookup_in_table(jur, class_number)
                if L:
                    return L
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


    # -- jurisdiction tables (law) ---------------------------------------
    # K schedules give each country a number range plus a form table ("KHD 1-5999 General (Table KDZ-KH2)",
    # "KKY 1-4999 Ukraine (Table KJ-KKZ1 modified)"); the printed captions live in the table, so a number
    # such as KHD2641 is table row 2641 shifted by the range start. Expanded at query time, no rebuild.
    _JUR_STOP = {"general", "table", "modified", "south", "north", "central", "east", "west", "america",
                 "asia", "europe", "africa", "and", "the", "of", "states", "provinces", "etc", "law", "other",
                 "countries", "regions", "cities", "communities", "republic", "kingdom", "united", "federal"}

    def jurisdictions(self) -> list[dict]:
        """[{section, node, table, start, end, base, name, tokens}] for every range node that cites a numbered table."""
        if getattr(self, "_jur", None) is not None:
            return self._jur
        out = []
        rows = self.db.execute("SELECT * FROM nodes WHERE section LIKE 'K%' AND number LIKE '%-%' AND number NOT LIKE '%.%' "
                               "AND caption LIKE '%(Table K%' ORDER BY section, ord").fetchall()
        for r in rows:
            m = re.match(r"^(\d+)-(\d+)$", r["number"] or "")
            code = re.search(r"\(Table\s+([A-Z][A-Za-z0-9\-]*)", r["caption"])
            if not m or not code:
                continue
            t = self.table(code.group(1))
            if not t:
                continue
            nums = [int(re.match(r"^\d+", x["number"]).group(0)) for x in t.rows if x.get("number") and re.match(r"^\d+", x["number"])]
            if not nums:
                continue
            node = self._node(r)
            name = re.sub(r"\s*\(Table[^)]*\)", "", node.caption).strip()
            if name.lower().startswith("general") and node.parent is not None:
                name = self.get(node.parent).caption
            name = name.rsplit(":", 1)[-1].strip()           # "South America: Brazil" -> "Brazil"
            tokens = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", name)} - self._JUR_STOP
            out.append({"section": node.section, "node": node, "table": t, "start": int(m.group(1)), "end": int(m.group(2)),
                        "base": max(int(m.group(1)) - 1, 0), "name": name, "tokens": tokens})   # '1-5999' -> +0, '6201-6299' -> +6200, '0-4890' -> +0
        self._jur = out
        return out

    @staticmethod
    def _table_paths(t: "Table") -> list[list[str]]:
        """Ancestor captions for every table row, from the rows' indentation levels."""
        paths, stack = [], []
        for r in t.rows:
            lvl = r.get("level") or 0
            stack = stack[:lvl]
            paths.append([c for c in stack])
            stack = stack + [r["caption"]]
        return paths

    def _expand_row(self, jur: dict, row: dict) -> str:
        """Table row number -> class number: ('KHD', base 0, '2642.P73') -> 'KHD2642.P73'."""
        m = re.match(r"^(\d+)(.*)$", row["number"] or "")
        return f"{jur['section']}{int(m.group(1)) + jur['base']}{m.group(2)}" if m else ""

    def _lookup_in_table(self, jur: dict, class_number: str) -> Lookup | None:
        m = re.match(r"^\s*[A-Z]{1,3}\s*(\d+)(.*)$", class_number.strip())
        if not m:
            return None
        n = int(m.group(1)) - jur["base"]
        rest = re.sub(r"\s.*$", "", m.group(2) or "")
        t, paths = jur["table"], self._table_paths(jur["table"])
        cands = [(i, r) for i, r in enumerate(t.rows) if r.get("number") and re.match(rf"^{n}(?![0-9])", r["number"])]
        if not cands:   # a range row such as '18-19'
            cands = [(i, r) for i, r in enumerate(t.rows) if r.get("number") and (m2 := re.match(r"^(\d+)-(\d+)$", r["number"]))
                     and int(m2.group(1)) <= n <= int(m2.group(2))]
        if not cands:
            return None
        exact = next(((i, r) for i, r in cands if r["number"] == f"{n}{rest}"), None)
        if exact is None:
            exact = next(((i, r) for i, r in cands if r["number"].split(".")[0] == str(n) and "." not in r["number"]), cands[0])
        i, r = exact
        mk = lambda k, rr: Node(-1 - k, t.schedule, jur["section"], "table-row", self._expand_row(jur, rr)[len(jur["section"]):]
                                if rr.get("number") else None, rr["caption"], rr.get("notes") or [], None, None, None, None, 0)
        node = mk(i, r)
        anc_rows = []
        lvl = r.get("level") or 0
        for j in range(i - 1, -1, -1):
            if (t.rows[j].get("level") or 0) < lvl:
                anc_rows.append(mk(j, t.rows[j])); lvl = t.rows[j].get("level") or 0
        kids = []
        for j in range(i + 1, min(i + 30, len(t.rows))):
            if (t.rows[j].get("level") or 0) <= (r.get("level") or 0):
                break
            kids.append(mk(j, t.rows[j]))
        sibs = [mk(j, t.rows[j]) for j in range(max(0, i - 6), min(len(t.rows), i + 7))
                if j != i and (t.rows[j].get("level") or 0) == (r.get("level") or 0)] if not kids else []
        anc = self.ancestors(jur["node"]) + [jur["node"]] + list(reversed(anc_rows))
        return Lookup(class_number, node, [], anc, kids, sibs, {}, via=f"Table {t.code} ({jur['name']})")

    def search_tables(self, query: str, jurs: list[dict], limit: int = 8) -> list[dict]:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2]
        hits = []
        for jur in jurs:
            t, paths = jur["table"], self._table_paths(jur["table"])
            for r, path in zip(t.rows, paths):
                if not r.get("number"):
                    continue
                cap, par, notes = r["caption"].lower(), " ".join(path).lower(), " ".join(r.get("notes") or []).lower()
                score = 0.0
                for w in words:
                    if w in jur["tokens"]:
                        continue
                    if re.search(rf"\b{re.escape(w)}", cap):
                        score += 4
                    if re.search(rf"\b{re.escape(w)}", par):
                        score += 6
                    if re.search(rf"\b{re.escape(w)}", notes):
                        score += 1
                if score:
                    hits.append({"class_number": self._expand_row(jur, r), "caption": r["caption"], "path": " -- ".join(path),
                                 "notes": " ".join(r.get("notes") or []), "via": f"Table {t.code} for {jur['name']}", "rank": -score})
        hits.sort(key=lambda h: h["rank"])
        return hits[:limit]

    # -- search ---------------------------------------------------------
    @staticmethod
    def _fts_query(q: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", q)
        return " OR ".join(f'"{w}"' for w in words if len(w) > 1) or '""'

    def search(self, query: str, limit: int = 20, section: str | None = None) -> list[dict]:
        """Full-text search over schedule captions (with ancestor path and notes) and the
        schedules' own index terms. Returns [{class_number, caption, path, notes, via}] ranked.
        The ancestor path outweighs the leaf caption: many numbers have a generic leaf ("General
        works", "Relation to literature") with the topic words in the parents."""
        q = self._fts_query(query)
        out, seen = [], set()
        sql = ("SELECT node_id, number, caption, path, notes, bm25(caption_fts, 0, 0, 0, 4.0, 6.0, 1.0) AS r FROM caption_fts "
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
        jurs = []
        if section:
            jurs = [j for j in self.jurisdictions() if j["section"] == section.upper()]
        else:
            qwords = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)}
            jurs = [j for j in self.jurisdictions()
                    if any(qw.startswith(tok[:5]) and len(tok) >= 5 or qw == tok for tok in j["tokens"] for qw in qwords)]
        table_hits = self.search_tables(query, jurs[:6], limit=max(4, limit // 2)) if jurs else []
        sem_nodes, sem_tables = self._semantic_hits(query, limit, section, jurs)
        # hybrid: alternate full-text and semantic node hits, then the jurisdiction-table hits
        merged, used = [], set()
        a, b = list(out), list(sem_nodes)
        while (a or b) and len(merged) < limit:
            for src in (a, b):
                while src:
                    h = src.pop(0)
                    if h["class_number"] not in used:
                        used.add(h["class_number"]); merged.append(h); break
                if len(merged) >= limit:
                    break
        tabs = [h for h in table_hits + sem_tables if h["class_number"] not in used]
        if tabs:
            room = max(len(tabs[: max(4, limit // 2)]), 0)
            merged = merged[: max(limit - room, 0)] + tabs[:room]
        return merged[:limit]

    def _semantic_hits(self, query: str, limit: int, section: str | None, jurs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Vector hits from the S3 Vectors index (empty when unavailable): caption nodes, and rows of the
        jurisdiction tables in play, expanded to class numbers."""
        try:
            from .semantic import get_semantic
            sem = get_semantic()
        except Exception:  # noqa: BLE001
            return [], []
        nodes = [h for h in sem.search(query, max(8, limit), section=section) if h.get("kind") == "node" and h["class_number"]]
        tables = []
        for jur in jurs[:3]:
            for h in sem.search(query, 4, table=jur["table"].code):
                try:
                    row = jur["table"].rows[int(h["row"])]
                except (ValueError, IndexError, TypeError):
                    continue
                cn = self._expand_row(jur, row)
                if cn:
                    tables.append({**h, "class_number": cn, "via": f"semantic: Table {jur['table'].code} for {jur['name']}"})
        return nodes, tables

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
                lines.append(f"      ({h['via']}" + (f", similarity {h['score']:.2f}" if h.get("score") is not None else "") + ")")
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
