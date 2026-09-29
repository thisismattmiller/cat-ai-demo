"""Orchestrator: resource metadata -> class number (schedule-first classification) ->
book number + date (CSM shelflisting) -> validated call number."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from . import callnum, rules
from .classifier import Classification, Classifier
from .compose import Composer
from .llm import Provider, get_provider
from .profile import profile_resource
from .resource import Resource
from .schedule import get_schedules, schedule_tags
from .shelflist_api import ShelflistClient


@dataclass
class Result:
    class_number: str | None
    book_number: str | None          # what LC puts in 050 $b: Cutter(s) + date (+ volume etc.)
    call_number: str | None
    classification: dict | None      # None when the class number was supplied
    profile: dict | None
    cards: list[str]
    compose: dict | None
    schedule: str | None
    timing: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    resource: dict | None = None
    model: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)

    @property
    def marc_050(self) -> tuple[str, str] | None:
        if not self.call_number:
            return None
        return callnum.parse(self.call_number).marc_050()


class Pipeline:
    def __init__(self, model: str | None = None, effort: str | None = None, *, provider: Provider | None = None,
                 use_lambda: bool = True, shelflist: ShelflistClient | None = None):
        self.provider = provider or get_provider(model, effort)
        self.shelflist = shelflist or ShelflistClient()
        self.classifier = Classifier(self.provider, self.shelflist, use_lambda=use_lambda)
        self.composer = Composer(self.provider, self.shelflist)

    def run(self, resource: Resource, class_number: str | None = None, *, extra_context: str | None = None,
            exclude_bibids: set[str] | None = None, on_stage=None) -> Result:
        """Classify (unless class_number given), then shelflist. `exclude_bibids` hides
        records from the shelflist (the record under evaluation). `on_stage(name, info)`
        is called as each stage completes (classified / profiled / composed)."""
        self.shelflist.exclude_bibids = set(exclude_bibids or ())
        stage = on_stage or (lambda name, info: None)
        timing, usage = {}, {}
        classification: Classification | None = None
        t0 = time.time()
        if class_number is None:
            classification = self.classifier.classify(resource, extra_context=extra_context)
            timing["classify_s"] = round(time.time() - t0, 1)
            usage["classify"] = classification.usage
            class_number = classification.class_number
            stage("classified", {"class_number": class_number, "confidence": classification.confidence,
                                 "caption_path": classification.caption_path,
                                 "subject_headings": list(classification.subject_headings or [])})
            if not class_number:
                return Result(None, None, None, asdict(classification), None, [], None, None, timing, usage,
                              asdict(resource), getattr(self.provider, "model", None))
            if classification.subject_headings and not resource.subjects:
                resource.subjects = list(classification.subject_headings)
        class_number = class_number.strip().rstrip("+").strip()

        text = resource.text()
        sch = get_schedules()
        lookup = sch.lookup(class_number) if sch else None
        schedule_text = lookup.text() if lookup and lookup.narrowest else None

        t1 = time.time()
        prof, pu = profile_resource(text, class_number, self.provider, extra_context=extra_context, schedule_text=schedule_text)
        timing["profile_s"] = round(time.time() - t1, 1)
        usage["profile"] = pu
        tags = set(prof.tags) | (schedule_tags(lookup) if lookup else set())
        prof.tags = sorted(tags)
        cards = rules.route(tags, class_number)
        stage("profiled", {"tags": prof.tags, "cards": [c.id for c in cards]})

        t2 = time.time()
        cr = self.composer.compose(text, class_number, prof, cards=cards, extra_context=extra_context, schedule_text=schedule_text)
        timing["compose_s"] = round(time.time() - t2, 1)
        usage["compose"] = cr.usage

        cn = book = None
        if cr.call_number:
            parsed = callnum.try_parse(cr.call_number)
            if parsed:
                cn = parsed.format()
                book = parsed.marc_050()[1]
            else:
                cn = cr.call_number
        return Result(class_number=class_number, book_number=book, call_number=cn,
                      classification=asdict(classification) if classification else None,
                      profile=prof.model_dump(), cards=[c.id for c in cards],
                      compose={k: v for k, v in asdict(cr).items() if k != "cards"}, schedule=schedule_text,
                      timing=timing, usage=usage, resource=asdict(resource), model=getattr(self.provider, "model", None))
