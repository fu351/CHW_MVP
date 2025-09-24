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
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_event(parent, kind: str, name: str) -> Tuple[ET.Element, str]:
    eid = _sid(kind.capitalize())
    ev = ET.SubElement(parent, f"{{{BPMN_NS}}}{kind}Event", id=eid, name=name)
    return ev, eid


def _make_task(parent, name: str) -> Tuple[ET.Element, str]:
    tid = _sid("Task")
    t = ET.SubElement(parent, f"{{{BPMN_NS}}}task", id=tid, name=name)
    return t, tid


def _make_user_task(parent, name: str) -> Tuple[ET.Element, str]:
    tid = _sid("UserTask")
    t = ET.SubElement(parent, f"{{{BPMN_NS}}}userTask", id=tid, name=name)
    return t, tid


def _make_business_rule_task(parent, name: str, decision_ref: str = None) -> Tuple[ET.Element, str]:
    tid = _sid("BusinessRuleTask")
    t = ET.SubElement(parent, f"{{{BPMN_NS}}}businessRuleTask", id=tid, name=name)
    if decision_ref:
        t.set("implementation", "dmn")
        t.set("decisionRef", decision_ref)
    return t, tid


def _make_gateway(parent, name: str) -> Tuple[ET.Element, str]:
    gid = _sid("Gateway")
    g = ET.SubElement(parent, f"{{{BPMN_NS}}}exclusiveGateway", id=gid, name=name)
    return g, gid


def _make_parallel_gateway(parent, name: str) -> Tuple[ET.Element, str]:
    gid = _sid("ParGateway")
    g = ET.SubElement(parent, f"{{{BPMN_NS}}}parallelGateway", id=gid, name=name)
    return g, gid


def _make_flow(parent, src_id: str, tgt_id: str, name: str, condition: str = None) -> ET.Element:
    fid = _sid("Flow")
    flow = ET.SubElement(
        parent,
        f"{{{BPMN_NS}}}sequenceFlow",
        id=fid,
        name=name,
        sourceRef=src_id,
        targetRef=tgt_id,
    )
    
    # Add condition expression if provided
    if condition:
        cond_expr = ET.SubElement(
            flow,
            f"{{{BPMN_NS}}}conditionExpression",
            attrib={"{http://www.w3.org/2001/XMLSchema-instance}type": "tFormalExpression"},
        )
        cond_expr.text = condition
    
    # Add incoming/outgoing elements to source and target nodes
    # Find source and target elements in parent
    for elem in parent.iter():
        if elem.get('id') == src_id:
            outgoing = ET.SubElement(elem, f"{{{BPMN_NS}}}outgoing")
            outgoing.text = fid
        elif elem.get('id') == tgt_id:
            incoming = ET.SubElement(elem, f"{{{BPMN_NS}}}incoming")
            incoming.text = fid
    
    return flow


def build_bpmn(rulepacks: Dict[str, Iterable]) -> str:
    """Build a proper sequential clinical workflow BPMN instead of parallel chaos."""
    defs = ET.Element(
        f"{{{BPMN_NS}}}definitions",
        attrib={
            "id": _sid("Defs"), 
            "targetNamespace": "http://chatchw.local/bpmn"
        },
    )
    proc = ET.SubElement(defs, f"{{{BPMN_NS}}}process", id=_sid("Process"), isExecutable="true")

    # Create sequential clinical workflow
    start, start_id = _make_event(proc, "start", "Start Assessment")
    
    # Step 1: Collect basic patient information
    collect_info, collect_info_id = _make_user_task(proc, "Collect Patient Information")
    _make_flow(proc, start_id, collect_info_id, "begin_assessment")
    
    # Step 2: Assess danger signs first (highest priority rules)
    assess_danger, assess_danger_id = _make_business_rule_task(proc, "Assess Danger Signs", "danger_signs_decision")
    _make_flow(proc, collect_info_id, assess_danger_id, "check_danger_signs")
    
    # Step 3: Danger signs decision gateway
    danger_gw, danger_gw_id = _make_gateway(proc, "Danger Signs Present?")
    _make_flow(proc, assess_danger_id, danger_gw_id, "danger_assessment_complete")
    
    # Step 4: Collect specific symptoms if no immediate danger
    collect_symptoms, collect_symptoms_id = _make_user_task(proc, "Collect Specific Symptoms")
    _make_flow(proc, danger_gw_id, collect_symptoms_id, "no_immediate_danger", "danger_sign != true")
    
    # Step 5: Apply clinical rules for symptom assessment
    apply_rules, apply_rules_id = _make_business_rule_task(proc, "Apply Clinical Rules", "clinical_assessment")
    _make_flow(proc, collect_symptoms_id, apply_rules_id, "assess_symptoms")
    
    # Step 6: Final triage decision gateway
    triage_gw, triage_gw_id = _make_gateway(proc, "Triage Decision")
    _make_flow(proc, apply_rules_id, triage_gw_id, "make_triage_decision")
    
    # Create end events
    end_hospital, end_hospital_id = _make_event(proc, "end", "🏥 Hospital Referral")
    end_clinic, end_clinic_id = _make_event(proc, "end", "🏪 Clinic Referral") 
    end_home, end_home_id = _make_event(proc, "end", "🏠 Home Care")
    
    # Direct danger sign route to hospital
    _make_flow(proc, danger_gw_id, end_hospital_id, "immediate_hospital_referral", "danger_sign == true")
    
    # Triage decision routes
    _make_flow(proc, triage_gw_id, end_hospital_id, "to_hospital", "proposed_triage == 'hospital'")
    _make_flow(proc, triage_gw_id, end_clinic_id, "to_clinic", "proposed_triage == 'clinic'")
    
    # Default route to home care
    default_flow = _make_flow(proc, triage_gw_id, end_home_id, "to_home")
    triage_gw.set("default", default_flow.get("id"))
    danger_gw.set("default", _make_flow(proc, danger_gw_id, collect_symptoms_id, "continue_assessment").get("id"))

    xml = ET.tostring(defs, encoding="utf-8", xml_declaration=True)
    return xml.decode("utf-8")
