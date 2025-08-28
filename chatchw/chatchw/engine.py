from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Tuple

from rfc3339_validator import validate_rfc3339

from .resolver import resolve_triage
from .schema import (
    Action,
    AllOfCondition,
    AnyOfCondition,
    Decision,
    EncounterInput,
    ObservationCondition,
    Rule,
    SymCondition,
    TriageLevel,
    TraceEntry,
)


def _timestamp_rfc3339() -> str:
    ts = datetime.now(timezone.utc).isoformat()
    if "." in ts:
        ts = ts.split(".")[0] + "+00:00"
    assert validate_rfc3339(ts)
    return ts


def _obs_map(enc: EncounterInput) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for o in enc.observations:
        m[o.id] = o.value
    return m


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
        if v is None:
            return False
        return _compare(cond.op, v, float(cond.value))
    if isinstance(cond, SymCondition):
        sv = getattr(enc.symptoms, cond.sym)
        return sv == cond.eq
    if isinstance(cond, AnyOfCondition):
        return any(eval_condition(c, enc, flags) for c in cond.any_of)
    if isinstance(cond, AllOfCondition):
        return all(eval_condition(c, enc, flags) for c in cond.all_of)
    raise TypeError("unknown condition type")


def run_rules(rules: Iterable[Rule], enc: EncounterInput) -> Tuple[Dict[str, bool], List[str], List[Action], List[TriageLevel], List[TraceEntry]]:
    flags: Dict[str, bool] = {}
    reasons: List[str] = []
    actions: List[Action] = []
    proposed: List[TriageLevel] = []
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
            if r.then.actions:
                for act in r.then.actions:
                    actions.append(act)
            if r.then.propose_triage:
                proposed.append(r.then.propose_triage)
            trace.append(
                TraceEntry(
                    rule_id=r.rule_id,
                    guideline_ref=r.then.guideline_ref,
                    timestamp=_timestamp_rfc3339(),
                )
            )
    return flags, reasons, actions, proposed, trace


def decide(enc: EncounterInput, rulepacks: Mapping[str, Iterable[Rule]]) -> Decision:
    all_flags: Dict[str, bool] = {}
    all_reasons: List[str] = []
    all_actions: List[Action] = []
    proposed: List[TriageLevel] = []
    trace: List[TraceEntry] = []

    for _module, rules in rulepacks.items():
        flags, reasons, actions, prop, tr = run_rules(rules, enc)
        for k, v in flags.items():
            if v:
                all_flags[k] = True
        for r in reasons:
            if r not in all_reasons:
                all_reasons.append(r)
        all_actions.extend(actions)
        proposed.extend(prop)
        trace.extend(tr)

    triage = resolve_triage(all_flags, proposed, enc.context)
    return Decision(triage=triage, actions=all_actions, reasons=all_reasons, trace=trace)

