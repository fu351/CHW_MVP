from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional

from .schema import (
    AllOfCondition,
    AnyOfCondition,
    ObservationCondition,
    SymCondition,
)


DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"
ET.register_namespace("dmn", DMN_NS)


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _feel_for_condition(c) -> str:
    # Handle dict conditions (from JSON)
    if isinstance(c, dict):
        if 'obs' in c:
            op_map = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
            return f"{c['obs']} {op_map[c['op']]} {c['value']}"
        elif 'sym' in c:
            # Handle both dictionary and object access for 'eq' field
            eq_value = None
            if hasattr(c, 'get'):
                eq_value = c.get('eq', True)  # Default to True if missing
            elif hasattr(c, 'eq'):
                eq_value = c.eq
            else:
                eq_value = True  # Default fallback
            
            val = "true" if bool(eq_value) else "false"
            if isinstance(eq_value, (int, float)):
                val = str(eq_value)
            return f"{c['sym']} = {val}"
        elif 'any_of' in c:
            return "(" + " or ".join(_feel_for_condition(x) for x in c['any_of']) + ")"
        elif 'all_of' in c:
            return "(" + " and ".join(_feel_for_condition(x) for x in c['all_of']) + ")"
    # Handle object conditions (from Rule objects)
    else:
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


def _collect_inputs_for_rule(r) -> List[str]:
    inputs: set[str] = set()
    when_conditions = r.when if hasattr(r, 'when') else r.get('when', [])
    
    for c in when_conditions:
        # Handle dict conditions (from JSON)
        if isinstance(c, dict):
            if 'obs' in c:
                inputs.add(c['obs'])
            elif 'sym' in c:
                inputs.add(c['sym'])
            elif 'any_of' in c:
                # Create a temporary object for recursive call
                temp_rule = {'when': c['any_of']}
                for k in _collect_inputs_for_rule(temp_rule):
                    inputs.add(k)
            elif 'all_of' in c:
                # Create a temporary object for recursive call
                temp_rule = {'when': c['all_of']}
                for k in _collect_inputs_for_rule(temp_rule):
                    inputs.add(k)
        # Handle object conditions (from Rule objects)
        else:
            if isinstance(c, ObservationCondition):
                inputs.add(c.obs)
            elif isinstance(c, SymCondition):
                inputs.add(c.sym)
            elif isinstance(c, AnyOfCondition):
                temp_rule = {'when': c.any_of}
                for k in _collect_inputs_for_rule(temp_rule):
                    inputs.add(k)
            elif isinstance(c, AllOfCondition):
                temp_rule = {'when': c.all_of}
                for k in _collect_inputs_for_rule(temp_rule):
                    inputs.add(k)
    return sorted(inputs)


def _infer_var_type_from_condition(cond) -> Optional[str]:
    """Infer FEEL typeRef from a single condition-like object/dict.
    Returns one of {"number", "boolean"} or None when unknown.
    """
    # Dict-based
    if isinstance(cond, dict):
        if 'obs' in cond:
            return "number"
        if 'sym' in cond:
            return "boolean"
        if 'any_of' in cond:
            for sub in cond['any_of']:
                t = _infer_var_type_from_condition(sub)
                if t:
                    return t
        if 'all_of' in cond:
            for sub in cond['all_of']:
                t = _infer_var_type_from_condition(sub)
                if t:
                    return t
        return None
    # Object-based
    from .schema import ObservationCondition as _Obs, SymCondition as _Sym, AnyOfCondition as _Any, AllOfCondition as _All
    if isinstance(cond, _Obs):
        return "number"
    if isinstance(cond, _Sym):
        return "boolean"
    if isinstance(cond, _Any):
        for sub in cond.any_of:
            t = _infer_var_type_from_condition(sub)
            if t:
                return t
    if isinstance(cond, _All):
        for sub in cond.all_of:
            t = _infer_var_type_from_condition(sub)
            if t:
                return t
    return None


def _infer_variable_types(rulepacks: Dict[str, Iterable]) -> Dict[str, str]:
    """Infer FEEL typeRef for each input variable across all rules.
    - Variables appearing in observation comparisons -> number
    - Variables appearing as boolean symbols -> boolean
    - Otherwise -> string
    """
    var_types: Dict[str, str] = {}
    # Seed with seen variables
    for rules in rulepacks.values():
        for r in rules:
            when_conditions = r.when if hasattr(r, 'when') else r.get('when', [])
            for cond in when_conditions:
                for var_name in _collect_inputs_for_rule({'when': [cond]}):
                    if var_name:
                        t = _infer_var_type_from_condition(cond)
                        # keep existing or leave undecided
                        if var_name not in var_types:
                            var_types[var_name] = 'string'
                    else:
                        # Prefer number over boolean over string if conflicting later
                        current = var_types.get(var_name)
                        if current is None or (current == 'string') or (current == 'boolean' and t == 'number'):
                            var_types[var_name] = t
    # Any variables not seen in conditions default to string
    # Collect all variables
    seen_vars: set[str] = set()
    for rules in rulepacks.values():
        for r in rules:
            for v in _collect_inputs_for_rule(r):
                seen_vars.add(v)
    for v in seen_vars:
        var_types.setdefault(v, 'string')
    return var_types


def _priority_of(rule_obj) -> int:
    """Get priority of a rule object."""
    if hasattr(rule_obj, 'priority'):
        try:
            return int(getattr(rule_obj, 'priority') or 0)
        except Exception:
            return 0
    if isinstance(rule_obj, dict):
        try:
            return int(rule_obj.get('then', {}).get('priority', rule_obj.get('priority', 0)) or 0)
        except Exception:
            return 0
    return 0


def _separate_rules_by_priority(rulepacks: Dict[str, Iterable]) -> Dict[str, List]:
    """Separate rules into danger signs (priority 100) and clinical assessment (lower priority)."""
    danger_rules = []
    clinical_rules = []
    
    for rules in rulepacks.values():
        for rule in rules:
            priority = _priority_of(rule)
            if priority >= 100:  # Danger signs
                danger_rules.append(rule)
            else:  # Clinical assessment
                clinical_rules.append(rule)
    
    return {
        'danger_signs': sorted(danger_rules, key=_priority_of, reverse=True),
        'clinical_assessment': sorted(clinical_rules, key=_priority_of, reverse=True)
    }


def _create_decision_table(parent, decision_id: str, decision_name: str, rules: List, var_types: Dict[str, str]) -> tuple:
    """Create a single decision table for a group of rules."""
    dec = ET.SubElement(parent, f"{{{DMN_NS}}}decision", attrib={"id": decision_id, "name": decision_name})
    table = ET.SubElement(dec, f"{{{DMN_NS}}}decisionTable", attrib={"id": _sid("Table"), "hitPolicy": "FIRST"})
    
    # Collect only variables used by these specific rules
    columns = sorted({v for r in rules for v in _collect_inputs_for_rule(r)})
    
    # Create input columns
    for v in columns:
        inp = ET.SubElement(table, f"{{{DMN_NS}}}input", attrib={"id": _sid("Input")})
        inp_expr = ET.SubElement(inp, f"{{{DMN_NS}}}inputExpression", attrib={"id": _sid("Expr"), "typeRef": var_types.get(v, 'string')})
        ET.SubElement(inp_expr, f"{{{DMN_NS}}}text").text = v
    
    # Create output column
    ET.SubElement(table, f"{{{DMN_NS}}}output", attrib={"id": _sid("Output"), "name": "result", "typeRef": "string"})
    
    return dec, table, columns


def _get_feel_operator(op: str) -> str:
    """Convert operation to FEEL operator."""
    op_map = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    return op_map.get(op, "=")


def _generate_output_value(then_clause, output_key: str) -> str:
    """Generate the output value for a rule based on its then clause."""
    if output_key == "danger_sign":
        # For danger signs, output true if this rule triggers
        return "true"
    
    # For clinical assessment, extract the proposed triage
    if hasattr(then_clause, 'propose_triage'):
        return then_clause.propose_triage or "home"
    elif isinstance(then_clause, dict):
        return then_clause.get('propose_triage', 'home')
    
    return "home"


def _add_rule_to_table(table: ET.Element, rule, columns: List[str], output_key: str):
    """Add a single rule to a decision table."""
    rule_id = rule.rule_id if hasattr(rule, 'rule_id') else rule.get('rule_id', 'unknown')
    when_conditions = rule.when if hasattr(rule, 'when') else rule.get('when', [])
    then_clause = rule.then if hasattr(rule, 'then') else rule.get('then', {})
    
    rule_el = ET.SubElement(table, f"{{{DMN_NS}}}rule", attrib={"id": f"rule_{rule_id}"})
    
    # Create condition map for this rule's variables
    condition_map = {}
    for cond in when_conditions:
        if isinstance(cond, dict):
            if 'obs' in cond:
                condition_map[cond['obs']] = f"{cond['obs']} {_get_feel_operator(cond['op'])} {cond['value']}"
            elif 'sym' in cond:
                eq_val = cond.get('eq', True)
                val_str = "true" if bool(eq_val) else "false"
                if isinstance(eq_val, (int, float)):
                    val_str = str(eq_val)
                condition_map[cond['sym']] = f"{cond['sym']} = {val_str}"
    
    # Generate input entries for all columns in this table
    for v in columns:
        ie = ET.SubElement(rule_el, f"{{{DMN_NS}}}inputEntry", attrib={"id": _sid("InEntry")})
        condition_text = condition_map.get(v, "-")
        ET.SubElement(ie, f"{{{DMN_NS}}}text").text = condition_text
    
    # Generate output entry
    output_value = _generate_output_value(then_clause, output_key)
    out_entry = ET.SubElement(rule_el, f"{{{DMN_NS}}}outputEntry", attrib={"id": _sid("OutEntry")})
    ET.SubElement(out_entry, f"{{{DMN_NS}}}text").text = output_value


def generate_dmn(rulepacks: Dict[str, Iterable]) -> str:
    """Generate hierarchical DMN with separate decision tables for danger signs and clinical assessment."""
    defs = ET.Element(
        f"{{{DMN_NS}}}definitions",
        attrib={
            "id": _sid("Defs"),
            "name": "ChatCHW Clinical Decision Support",
            "namespace": "chatchw",
            "exporter": "ChatCHW",
            "exporterVersion": "2.0.0",
            "dmnConformanceLevel": "ConformanceLevel-3",
        },
    )

    # Separate rules by priority for hierarchical processing
    separated_rules = _separate_rules_by_priority(rulepacks)
    
    # Determine input variables and types for all rules
    input_vars: set[str] = set()
    for rule_group in separated_rules.values():
        for r in rule_group:
            for v in _collect_inputs_for_rule(r):
                input_vars.add(v)
    var_types = _infer_variable_types({"all": [r for group in separated_rules.values() for r in group]})

    # Create input data elements
    for v in sorted(input_vars):
        inp = ET.SubElement(defs, f"{{{DMN_NS}}}inputData", attrib={"id": f"input_{v}", "name": v})
        ET.SubElement(inp, f"{{{DMN_NS}}}variable", attrib={"name": v, "typeRef": var_types.get(v, 'string')})

    # Create Danger Signs Decision Table (Priority 100+ rules)
    if separated_rules['danger_signs']:
        danger_dec, danger_table, danger_columns = _create_decision_table(
            defs, "danger_signs_decision", "Danger Signs Assessment", 
            separated_rules['danger_signs'], var_types
        )
        
        # Add rules to danger signs table
        for r in separated_rules['danger_signs']:
            _add_rule_to_table(danger_table, r, danger_columns, "danger_sign")
        
        # Add information requirements
        for v in danger_columns:
            ir = ET.SubElement(danger_dec, f"{{{DMN_NS}}}informationRequirement", attrib={"id": _sid("IR")})
            ET.SubElement(ir, f"{{{DMN_NS}}}requiredInput", attrib={"href": f"#input_{v}"})
    
    # Create Clinical Assessment Decision Table (Lower priority rules)
    if separated_rules['clinical_assessment']:
        clinical_dec, clinical_table, clinical_columns = _create_decision_table(
            defs, "clinical_assessment", "Clinical Assessment", 
            separated_rules['clinical_assessment'], var_types
        )
        
        # Add rules to clinical assessment table
        for r in separated_rules['clinical_assessment']:
            _add_rule_to_table(clinical_table, r, clinical_columns, "proposed_triage")
        
        # Add information requirements
        for v in clinical_columns:
            ir = ET.SubElement(clinical_dec, f"{{{DMN_NS}}}informationRequirement", attrib={"id": _sid("IR")})
            ET.SubElement(ir, f"{{{DMN_NS}}}requiredInput", attrib={"href": f"#input_{v}"})

    xml = ET.tostring(defs, encoding="utf-8", xml_declaration=True)
    return xml.decode("utf-8")