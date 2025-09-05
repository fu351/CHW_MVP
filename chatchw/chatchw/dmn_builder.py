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
            # Walk each condition and record types for referenced variables
            for cond in when_conditions:
                t = _infer_var_type_from_condition(cond)
                # Figure out which var name this condition references
                var_name: Optional[str] = None
                if isinstance(cond, dict):
                    if 'obs' in cond:
                        var_name = cond['obs']
                    elif 'sym' in cond:
                        var_name = cond['sym']
                else:
                    if hasattr(cond, 'obs') and cond.obs:
                        var_name = cond.obs
                    elif hasattr(cond, 'sym') and cond.sym:
                        var_name = cond.sym
                if var_name:
                    if t is None:
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


def generate_dmn(rulepacks: Dict[str, Iterable]) -> str:
    defs = ET.Element(
        f"{{{DMN_NS}}}definitions",
        attrib={
            "id": _sid("Defs"),
            "name": "ChatCHW",
            "namespace": "chatchw",
            # DMN 1.4 exporter metadata and conformance declaration (informative)
            "exporter": "ChatCHW",
            "exporterVersion": "0.1.0",
            "dmnConformanceLevel": "ConformanceLevel-2",
        },
    )

    # Determine input variables and types
    input_vars: set[str] = set()
    for rules in rulepacks.values():
        for r in rules:
            for v in _collect_inputs_for_rule(r):
                input_vars.add(v)
    var_types = _infer_variable_types(rulepacks)

    # Emit inputData with variable/typeRef
    for v in sorted(input_vars):
        inp = ET.SubElement(defs, f"{{{DMN_NS}}}inputData", attrib={"id": f"input_{v}", "name": v})
        ET.SubElement(inp, f"{{{DMN_NS}}}variable", attrib={"name": v, "typeRef": var_types.get(v, 'string')})

    for module_name, rules in rulepacks.items():
        dec = ET.SubElement(defs, f"{{{DMN_NS}}}decision", attrib={"id": f"decision_{module_name}", "name": f"{module_name}_decision"})
        table = ET.SubElement(dec, f"{{{DMN_NS}}}decisionTable", attrib={"id": _sid("Table"), "hitPolicy": "FIRST"})
        columns = sorted({v for r in rules for v in _collect_inputs_for_rule(r)})
        for v in columns:
            inp = ET.SubElement(table, f"{{{DMN_NS}}}input", attrib={"id": _sid("Input")})
            inp_expr = ET.SubElement(inp, f"{{{DMN_NS}}}inputExpression", attrib={"id": _sid("Expr"), "typeRef": var_types.get(v, 'string')})
            ET.SubElement(inp_expr, f"{{{DMN_NS}}}text").text = v
        ET.SubElement(table, f"{{{DMN_NS}}}output", attrib={"id": _sid("Output"), "name": "effect", "typeRef": "string"})

        # Sort rules by priority (handle both Rule objects and plain dict rules)
        def _priority_of(rule_obj) -> int:
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

        sorted_rules = sorted(list(rules), key=_priority_of, reverse=True)
        
        for r in sorted_rules:
            rule_id = r.rule_id if hasattr(r, 'rule_id') else r.get('rule_id', 'unknown')
            when_conditions = r.when if hasattr(r, 'when') else r.get('when', [])
            then_clause = r.then if hasattr(r, 'then') else r.get('then', {})
            
            rule_el = ET.SubElement(table, f"{{{DMN_NS}}}rule", attrib={"id": f"rule_{rule_id}"})
            
            # Create a map of input variables to their conditions
            condition_map = {}
            
            def evaluate_complex_condition_for_variable(var_name, when_conditions):
                """Evaluate if a variable has specific constraints in complex conditions."""
                def check_condition_tree(cond):
                    if isinstance(cond, dict):
                        if 'all_of' in cond:
                            # For ALL_OF, we need ALL conditions to be true
                            var_conditions = []
                            for sub_cond in cond['all_of']:
                                var_cond = check_condition_tree(sub_cond)
                                if var_cond:
                                    var_conditions.append(var_cond)
                            return var_conditions[0] if len(var_conditions) == 1 else None
                        elif 'any_of' in cond:
                            # For ANY_OF, check if this variable appears in any branch
                            for sub_cond in cond['any_of']:
                                var_cond = check_condition_tree(sub_cond)
                                if var_cond:
                                    return var_cond
                            return None
                        elif 'obs' in cond and cond['obs'] == var_name:
                            return _feel_for_condition(cond)
                        elif 'sym' in cond and cond['sym'] == var_name:
                            return _feel_for_condition(cond)
                    else:
                        # Handle Pydantic objects
                        if hasattr(cond, 'obs') and cond.obs == var_name:
                            return _feel_for_condition(cond)
                        elif hasattr(cond, 'sym') and cond.sym == var_name:
                            return _feel_for_condition(cond)
                    return None
                
                # Check if this variable has any specific conditions
                for c in when_conditions:
                    result = check_condition_tree(c)
                    if result:
                        return result
                return "-"  # Default to catch-all only if variable is not constrained
            
            # Fill each column appropriately
            for v in columns:
                ie = ET.SubElement(rule_el, f"{{{DMN_NS}}}inputEntry", attrib={"id": _sid("InEntry")})
                condition_text = evaluate_complex_condition_for_variable(v, when_conditions)
                ET.SubElement(ie, f"{{{DMN_NS}}}text").text = condition_text
            
            effect_parts = []
            # Handle both dict and object then clauses
            if hasattr(then_clause, 'propose_triage'):
                if then_clause.propose_triage:
                    triage_val = then_clause.propose_triage
                    effect_parts.append(f"triage:{triage_val}")
                    # Emit BPMN-alignment flags for triage to drive gateway conditions
                    if str(triage_val).lower() == 'clinic':
                        effect_parts.append("flag:clinic_referral")
                    elif str(triage_val).lower() == 'hospital':
                        effect_parts.append("flag:danger_sign")
                if then_clause.set_flags:
                    effect_parts.extend([f"flag:{f}" for f in then_clause.set_flags])
                if then_clause.reasons:
                    effect_parts.extend([f"reason:{x}" for x in then_clause.reasons])
            else:
                # Handle dict then clause
                if then_clause.get('propose_triage'):
                    triage_val = then_clause['propose_triage']
                    effect_parts.append(f"triage:{triage_val}")
                    # Emit BPMN-alignment flags for triage to drive gateway conditions
                    if str(triage_val).lower() == 'clinic':
                        effect_parts.append("flag:clinic_referral")
                    elif str(triage_val).lower() == 'hospital':
                        effect_parts.append("flag:danger_sign")
                if then_clause.get('set_flags'):
                    effect_parts.extend([f"flag:{f}" for f in then_clause['set_flags']])
                if then_clause.get('reasons'):
                    effect_parts.extend([f"reason:{x}" for x in then_clause['reasons']])
            
            out_entry = ET.SubElement(rule_el, f"{{{DMN_NS}}}outputEntry", attrib={"id": _sid("OutEntry")})
            ET.SubElement(out_entry, f"{{{DMN_NS}}}text").text = ", ".join(effect_parts) if effect_parts else "noop"

        # Note: No default catch-all rules needed for clinical guidelines
        # Clinical rules are designed to be specific and don't need 100% input space coverage

        for v in columns:
            ir = ET.SubElement(dec, f"{{{DMN_NS}}}informationRequirement", attrib={"id": _sid("IR")})
            ET.SubElement(ir, f"{{{DMN_NS}}}requiredInput", attrib={"href": f"#input_{v}"})

    xml = ET.tostring(defs, encoding="utf-8", xml_declaration=True)
    return xml.decode("utf-8")
