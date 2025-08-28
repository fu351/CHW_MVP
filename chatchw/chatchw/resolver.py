from __future__ import annotations

from typing import Iterable, Mapping

from .schema import ContextFlags, TriageLevel


def resolve_triage(flags: Mapping[str, bool], proposed: Iterable[TriageLevel], ctx: ContextFlags) -> TriageLevel:
    if flags.get("danger.sign", False):
        return "hospital"
    if flags.get("suspected.malaria", False) and ctx.malaria_present and (ctx.stockout or {}).get("antimalarial", False):
        return "hospital"
    proposed_set = set(proposed)
    if "hospital" in proposed_set:
        return "hospital"
    if "clinic" in proposed_set:
        return "clinic"
    return "home"

