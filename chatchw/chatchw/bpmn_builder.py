from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Tuple

from .schema import Rule


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DI_NS = "http://www.omg.org/spec/DD/20100524/DI"
DC_NS = "http://www.omg.org/spec/DD/20100524/DC"

ET.register_namespace("bpmn", BPMN_NS)
ET.register_namespace("bpmndi", BPMNDI_NS)
ET.register_namespace("di", DI_NS)
ET.register_namespace("dc", DC_NS)


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_task(parent, name: str) -> Tuple[ET.Element, str]:
    tid = _sid("Task")
    t = ET.SubElement(parent, f"{{{BPMN_NS}}}task", id=tid, name=name)
    return t, tid


def _make_gateway(parent, name: str) -> Tuple[ET.Element, str]:
    gid = _sid("Gateway")
    g = ET.SubElement(parent, f"{{{BPMN_NS}}}exclusiveGateway", id=gid, name=name)
    return g, gid


def _make_event(parent, kind: str, name: str) -> Tuple[ET.Element, str]:
    eid = _sid(kind.capitalize())
    ev = ET.SubElement(parent, f"{{{BPMN_NS}}}{kind}Event", id=eid, name=name)
    return ev, eid


def _make_flow(parent, src_id: str, tgt_id: str, name: str) -> ET.Element:
    fid = _sid("Flow")
    return ET.SubElement(
        parent,
        f"{{{BPMN_NS}}}sequenceFlow",
        id=fid,
        name=name,
        sourceRef=src_id,
        targetRef=tgt_id,
    )


def build_bpmn(rulepacks: Dict[str, Iterable[Rule]]) -> str:
    """
    Build a minimal BPMN XML string. One process with linearized modules.
    Each module is represented by a gateway and a series of tasks; flow names are rule_ids.
    """
    defs = ET.Element(
        f"{{{BPMN_NS}}}definitions",
        attrib={"id": _sid("Defs"), "targetNamespace": "http://chatchw.local/bpmn"},
    )
    proc = ET.SubElement(defs, f"{{{BPMN_NS}}}process", id=_sid("Process"), isExecutable="false")

    start, start_id = _make_event(proc, "start", "Start")
    prev_id = start_id

    for module_name, rules in rulepacks.items():
        gw, gw_id = _make_gateway(proc, f"{module_name} checks")
        _make_flow(proc, prev_id, gw_id, f"{module_name}.enter")
        prev_id = gw_id
        for r in rules:
            task, task_id = _make_task(proc, f"{module_name}:{r.rule_id}")
            _make_flow(proc, prev_id, task_id, r.rule_id)
            prev_id = task_id

    end, end_id = _make_event(proc, "end", "End")
    _make_flow(proc, prev_id, end_id, "complete")

    xml = ET.tostring(defs, encoding="utf-8", xml_declaration=True)
    return xml.decode("utf-8")

