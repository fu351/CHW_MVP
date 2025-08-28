from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List

from .schema import (
    AllOfCondition,
    AnyOfCondition,
    ObservationCondition,
    Rule,
    SymCondition,
)


DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"
DMNDI_NS = "https://www.omg.org/spec/DMN/20191111/DMNDI/"
DI_NS = "http://www.omg.org/spec/DMN/20180521/DI/"
DC_NS = "http://www.omg.org/spec/DMN/20180521/DC/"

ET.register_namespace("dmn", DMN_NS)
ET.register_namespace("dmndi", DMNDI_NS)
ET.register_namespace("di", DI_NS)
ET.register_namespace("dc", DC_NS)


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _feel_for_condition(c) -> str:
    if isinstance(c, ObservationCondition):
        op_map = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
        return f"{c.obs} {op_map[c.op]} {c.value}"
    if isinstance(c, SymCondition):
        val = "true" if bool(c.eq) else "false"
        if isinstance(c.eq, (int, float)):
            val = str(c.eq)
        return f"{c.sym} = {val}"
    if isinstance(c, AnyOfCondition):
        return "(" + " or ".join(_feel_for_condition(x) for x in c.any_of) + ")"
    if isinstance(c, AllOfCondition):
        return "(" + " and ".join(_feel_for_condition(x) for x in c.all_of) + ")"
    return ""


def _collect_inputs_for_rule(r: Rule) -> List[str]:
    inputs: set[str] = set()
    for c in r.when:
        if isinstance(c, ObservationCondition):
            inputs.add(c.obs)
        elif isinstance(c, SymCondition):
            inputs.add(c.sym)
        elif isinstance(c, AnyOfCondition):
            for k in _collect_inputs_for_rule(Rule(rule_id="tmp", when=c.any_of, then=r.then, priority=r.priority)).copy():  # type: ignore[arg-type]
                inputs.add(k)
        elif isinstance(c, AllOfCondition):
            for k in _collect_inputs_for_rule(Rule(rule_id="tmp", when=c.all_of, then=r.then, priority=r.priority)).copy():  # type: ignore[arg-type]
                inputs.add(k)
    return sorted(inputs)


def generate_dmn(rulepacks: Dict[str, Iterable[Rule]]) -> str:
    """
    Generate a DMN 1.3 XML with simple decision tables, one decision per module.
    """
    defs = ET.Element(f"{{{DMN_NS}}}definitions", attrib={"id": _sid("Defs"), "name": "ChatCHW", "namespace": "chatchw"})

    input_vars: set[str] = set()
    for rules in rulepacks.values():
        for r in rules:
            for v in _collect_inputs_for_rule(r):
                input_vars.add(v)

    for v in sorted(input_vars):
        ET.SubElement(defs, f"{{{DMN_NS}}}inputData", attrib={"id": f"input_{v}", "name": v})

    for module_name, rules in rulepacks.items():
        dec = ET.SubElement(defs, f"{{{DMN_NS}}}decision", attrib={"id": f"decision_{module_name}", "name": f"{module_name}_decision"})
        table = ET.SubElement(dec, f"{{{DMN_NS}}}decisionTable", attrib={"id": _sid("Table"), "hitPolicy": "PRIORITY"})
        columns = sorted({v for r in rules for v in _collect_inputs_for_rule(r)})
        for v in columns:
            inp = ET.SubElement(table, f"{{{DMN_NS}}}input", attrib={"id": _sid("Input")})
            ET.SubElement(inp, f"{{{DMN_NS}}}inputExpression", attrib={"id": _sid("Expr"), "typeRef": "string"}).text = v
        ET.SubElement(table, f"{{{DMN_NS}}}output", attrib={"id": _sid("Output"), "name": "effect", "typeRef": "string"})

        for r in sorted(list(rules), key=lambda x: x.priority, reverse=True):
            rule_el = ET.SubElement(table, f"{{{DMN_NS}}}rule", attrib={"id": f"rule_{r.rule_id}"})
            feel = " and ".join(_feel_for_condition(c) for c in r.when)
            used = {v for v in _collect_inputs_for_rule(r)}
            for v in columns:
                ie = ET.SubElement(rule_el, f"{{{DMN_NS}}}inputEntry", attrib={"id": _sid("InEntry")})
                ET.SubElement(ie, f"{{{DMN_NS}}}text").text = feel if v in used else "-"
            effect_parts = []
            if r.then.propose_triage:
                effect_parts.append(f"triage:{r.then.propose_triage}")
            if r.then.set_flags:
                effect_parts.extend([f"flag:{f}" for f in r.then.set_flags])
            if r.then.reasons:
                effect_parts.extend([f"reason:{x}" for x in r.then.reasons])
            out_entry = ET.SubElement(rule_el, f"{{{DMN_NS}}}outputEntry", attrib={"id": _sid("OutEntry")})
            ET.SubElement(out_entry, f"{{{DMN_NS}}}text").text = ", ".join(effect_parts) if effect_parts else "noop"

        for v in columns:
            ir = ET.SubElement(dec, f"{{{DMN_NS}}}informationRequirement", attrib={"id": _sid("IR")})
            ET.SubElement(ir, f"{{{DMN_NS}}}requiredInput", attrib={"href": f"#input_{v}"})

    xml = ET.tostring(defs, encoding="utf-8", xml_declaration=True)
    return xml.decode("utf-8")

