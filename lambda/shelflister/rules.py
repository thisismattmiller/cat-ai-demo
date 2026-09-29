"""Rule cards: load rules/*.md, and route a resource profile to the applicable cards."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"

# Cards that are useful for nearly every call number, in the order they should appear.
CORE_ORDER = ["G053", "G055", "G058", "G063", "G065", "G100", "G140"]   # G070 (050 formatting) is done in code


@dataclass
class Card:
    id: str
    title: str
    source: str
    always_load: bool
    triggers: list[str]
    class_prefixes: list[str]
    see_also: list[str]
    summary: str
    body: str
    path: Path
    extra: dict = field(default_factory=dict)

    def matches(self, tags: set[str], class_letters: str) -> bool:
        if self.always_load:
            return True
        tag_hit = bool(set(self.triggers) & tags) if self.triggers else False
        prefix_hit = any(class_letters.startswith(p) for p in self.class_prefixes) if self.class_prefixes else False
        if self.triggers and self.class_prefixes:
            # both given: the class prefix restricts (an artists card only in N; a
            # literary-author card only in P) and a tag must also fire
            return prefix_hit and tag_hit
        return tag_hit or prefix_hit

    def text(self) -> str:
        return f"<rule id=\"{self.id}\" title=\"{self.title}\" source=\"{self.source}\">\n{self.body.strip()}\n</rule>"


def _parse_card(path: Path) -> Card:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if not m:
        raise ValueError(f"{path}: missing frontmatter")
    meta = yaml.safe_load(m.group(1)) or {}
    return Card(
        id=str(meta.get("id", path.stem)),
        title=str(meta.get("title", "")),
        source=str(meta.get("source", "")),
        always_load=bool(meta.get("always_load", False)),
        triggers=[str(t) for t in (meta.get("triggers") or [])],
        class_prefixes=[str(p) for p in (meta.get("class_prefixes") or [])],
        see_also=[str(s) for s in (meta.get("see_also") or [])],
        summary=str(meta.get("summary", "")).strip(),
        body=m.group(2),
        path=path,
        extra={k: v for k, v in meta.items() if k not in {"id", "title", "source", "always_load", "triggers", "class_prefixes", "see_also", "summary"}},
    )


@lru_cache(maxsize=1)
def load_cards(rules_dir: Path = RULES_DIR) -> dict[str, Card]:
    cards = {}
    for p in sorted(rules_dir.glob("*.md")):
        if p.name == "SPEC.md":
            continue
        try:
            c = _parse_card(p)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"bad rule card {p}: {e}") from e
        cards[c.id] = c
    return cards


def route(tags: set[str], class_number: str, *, follow_see_also: bool = False) -> list[Card]:
    """Cards to load for a resource with these profile tags and this class number.
    Core cards first (fixed order), then conditional cards in id order."""
    cards = load_cards()
    letters = re.match(r"[A-Z]+", class_number.strip()).group(0) if re.match(r"[A-Z]+", class_number.strip()) else ""
    chosen: dict[str, Card] = {}
    for cid in CORE_ORDER:
        if cid in cards:
            chosen[cid] = cards[cid]
    for cid, c in cards.items():
        if cid not in chosen and c.matches(tags, letters):
            chosen[cid] = c
    if follow_see_also:
        for c in list(chosen.values()):
            for s in c.see_also:
                s = s.replace(" ", "")
                if s in cards and s not in chosen:
                    chosen[s] = cards[s]
    return list(chosen.values())


def index_text() -> str:
    """One line per card: id, title, summary. Given to the LLM so it can ask for a
    card that the router did not load."""
    return "\n".join(f"- {c.id} {c.title}: {c.summary}" for c in load_cards().values())


def short_index(exclude: set[str] | None = None) -> str:
    """One line: 'F010 General Principles of Classification; F060 Filing Rules; ...'."""
    exclude = exclude or set()
    return "; ".join(f"{c.id} {c.title}" for c in load_cards().values() if c.id not in exclude)


def all_triggers() -> set[str]:
    return {t for c in load_cards().values() for t in c.triggers}
