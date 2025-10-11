from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Tuple

from .schema import (
    AllOfCondition,
    AnyOfCondition,
    Decision,
    EncounterInput,
    ObservationCondition,
    Rule,
    SymCondition,
    TraceEntry,
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obs_map(enc: EncounterInput) -> Dict[str, float]:
    return {o.id: o.value for o in enc.observations}


def _compare(op: str, left: float, right: float) -> bool:
    if op == "eq":
        return float(left) == float(right)
    if op == "lt":
        return float(left) < float(right)
    if op == "le":
        return float(left) <= float(right)
    if op == "gt":
        return float(left) > float(right)
    if op == "ge":
        return float(left) >= float(right)
    raise ValueError(f"unknown op: {op}")


def eval_condition(cond, enc: EncounterInput, flags: Mapping[str, bool]) -> bool:
    if isinstance(cond, ObservationCondition):
        v = _obs_map(enc).get(cond.obs)
        return v is not None and _compare(cond.op, v, float(cond.value))
    if isinstance(cond, SymCondition):
        sv = getattr(enc.symptoms, cond.sym)
        return sv == cond.eq
    if isinstance(cond, AnyOfCondition):
        return any(eval_condition(c, enc, flags) for c in cond.any_of)
    if isinstance(cond, AllOfCondition):
        return all(eval_condition(c, enc, flags) for c in cond.all_of)
    raise TypeError("unknown condition type")


def run_rules(rules: Iterable[Rule], enc: EncounterInput) -> Tuple[Dict[str, bool], List[str], List[str], List[TraceEntry]]:
    flags: Dict[str, bool] = {}
    reasons: List[str] = []
    proposed: List[str] = []
    trace: List[TraceEntry] = []

    ordered = sorted(list(rules), key=lambda r: r.priority, reverse=True)
    for r in ordered:
        if all(eval_condition(c, enc, flags) for c in r.when):
            if r.then.set_flags:
                for f in r.then.set_flags:
                    flags[f] = True
            if r.then.reasons:
                for reason in r.then.reasons:
                    if reason not in reasons:
                        reasons.append(reason)
            if r.then.propose_triage:
                proposed.append(r.then.propose_triage)
            trace.append(TraceEntry(rule_id=r.rule_id, guideline_ref=r.then.guideline_ref, timestamp=_ts()))
    return flags, reasons, proposed, trace


def decide(enc: EncounterInput, rulepacks: Mapping[str, Iterable[Rule]]) -> Decision:
    all_flags: Dict[str, bool] = {}
    all_reasons: List[str] = []
    proposed: List[str] = []
    trace: List[TraceEntry] = []

    for _module, rules in rulepacks.items():
        flags, reasons, prop, tr = run_rules(rules, enc)
        for k, v in flags.items():
            if v:
                all_flags[k] = True
        for r in reasons:
            if r not in all_reasons:
                all_reasons.append(r)
        proposed.extend(prop)
        trace.extend(tr)

    if all_flags.get("danger.sign", False):
        triage = "hospital"
    elif "hospital" in set(proposed):
        triage = "hospital"
    elif "clinic" in set(proposed):
        triage = "clinic"
    else:
        triage = "home"

    return Decision(triage=triage, reasons=all_reasons, trace=trace)
