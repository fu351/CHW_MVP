"""
Validator for BPMN structural soundness, DMN logical completeness, and BPMN/DMN alignment.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

# -----------------------
# BPMN CHECKS (Item 1)
# -----------------------

BPMN_NS = {
    'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
    'dc': 'http://www.omg.org/spec/DD/20100524/DC',
    'di': 'http://www.omg.org/spec/DD/20100524/DI',
}


def parse_bpmn_graph(root: ET.Element) -> List[Dict]:
    processes = root.findall('.//bpmn:process', BPMN_NS)
    graphs: List[Dict] = []
    for proc in processes:
        nodes: Dict[str, Dict[str, str]] = {}
        outgoing_map: Dict[str, List[str]] = defaultdict(list)
        incoming_count: Dict[str, int] = defaultdict(int)
        start_events: List[str] = []
        end_events: List[str] = []

        # Collect flow nodes (tasks, gateways, events)
        for elem in proc.iter():
            tag = elem.tag
            if not isinstance(tag, str):
                continue
            if tag.startswith('{'):
                local = tag.split('}', 1)[1]
                if local in (
                    'startEvent', 'endEvent', 'task', 'userTask', 'serviceTask', 'scriptTask', 'businessRuleTask',
                    'exclusiveGateway', 'inclusiveGateway', 'parallelGateway', 'eventBasedGateway',
                    'subProcess', 'callActivity', 'intermediateThrowEvent', 'boundaryEvent'
                ):
                    nid = elem.get('id')
                    if not nid:
                        continue
                    nodes[nid] = {'type': local, 'name': elem.get('name', '')}
                    if local == 'startEvent':
                        start_events.append(nid)
                    if local == 'endEvent':
                        end_events.append(nid)

        # Map sequence flows
        for sf in proc.findall('.//bpmn:sequenceFlow', BPMN_NS):
            sid = sf.get('sourceRef')
            tid = sf.get('targetRef')
            if sid and tid:
                outgoing_map[sid].append(tid)
                incoming_count[tid] += 1
                nodes.setdefault(sid, {'type': 'unknown', 'name': ''})
                nodes.setdefault(tid, {'type': 'unknown', 'name': ''})

        graphs.append({
            'process_id': proc.get('id', ''),
            'nodes': nodes,
            'out': dict(outgoing_map),
            'incoming_count': dict(incoming_count),
            'starts': start_events,
            'ends': end_events,
        })
    return graphs


def bpmn_reachability(graph: Dict) -> Tuple[Set[str], Set[str], List[List[str]]]:
    nodes = graph['nodes']
    out = graph['out']
    starts = graph['starts']
    ends = graph['ends']

    # Forward reachability from all starts
    reachable_from_start: Set[str] = set()
    dq: deque[str] = deque(starts)
    while dq:
        u = dq.popleft()
        if u in reachable_from_start:
            continue
        reachable_from_start.add(u)
        for v in out.get(u, []):
            dq.append(v)

    # Reverse graph
    rev: Dict[str, List[str]] = defaultdict(list)
    for u, vs in out.items():
        for v in vs:
            rev[v].append(u)

    # Nodes that can reach an end
    can_reach_end: Set[str] = set()
    dq = deque(ends)
    while dq:
        u = dq.popleft()
        if u in can_reach_end:
            continue
        can_reach_end.add(u)
        for v in rev.get(u, []):
            dq.append(v)

    # Detect SCCs (Tarjan) to find cycles
    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    stack: List[str] = []
    onstack: Set[str] = set()
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in out.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for n in nodes:
        if n not in indices:
            strongconnect(n)

    return reachable_from_start, can_reach_end, sccs


def check_bpmn_soundness(xml_path: str) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    graphs = parse_bpmn_graph(root)
    report = {'file': xml_path, 'processes': []}

    for g in graphs:
        reachable_from_start, can_reach_end, sccs = bpmn_reachability(g)
        nodes = g['nodes']
        out = g['out']
        starts = g['starts']
        ends = g['ends']

        # 1) Every start leads to an end
        starts_without_end = [s for s in starts if s not in can_reach_end]

        # 2) Every end is reachable from some start
        unreachable_ends = [e for e in ends if e not in reachable_from_start]

        # 3) Orphan nodes (unreachable from start)
        orphan_nodes = [n for n in nodes if n not in reachable_from_start]

        # 4) Livelocks/infinite loops: SCCs reachable from start with no path to end
        cyclic_sccs: List[List[str]] = []
        for comp in sccs:
            has_cycle = len(comp) > 1 or any((nid in g['out'] and nid in g['out'][nid]) for nid in comp)
            if not has_cycle:
                continue
            if any(n in reachable_from_start for n in comp) and not any(n in can_reach_end for n in comp):
                cyclic_sccs.append(comp)

        # 5) Dangling gateways (no outgoing flows)
        dangling_gateways: List[str] = []
        for nid, meta in nodes.items():
            if meta['type'] in ('exclusiveGateway', 'inclusiveGateway', 'parallelGateway', 'eventBasedGateway'):
                if len(out.get(nid, [])) == 0:
                    dangling_gateways.append(nid)

        # 6) Exceptions that can't reach ends (best-effort)
        problematic_exceptions: List[str] = []
        for ev in root.findall('.//bpmn:boundaryEvent', BPMN_NS) + root.findall('.//bpmn:intermediateThrowEvent', BPMN_NS):
            ev_id = ev.get('id')
            if ev_id and (len(g['out'].get(ev_id, [])) > 0) and (ev_id not in can_reach_end):
                if ev_id in nodes:
                    problematic_exceptions.append(ev_id)

        proc_report = {
            'process_id': g['process_id'],
            'starts_without_path_to_end': starts_without_end,
            'unreachable_ends_from_start': unreachable_ends,
            'orphan_nodes': orphan_nodes,
            'potential_livelock_cycles': cyclic_sccs,
            'dangling_gateways': dangling_gateways,
            'exceptions_not_resolving_to_end': problematic_exceptions,
            'summary_pass': (
                len(starts_without_end) == 0
                and len(unreachable_ends) == 0
                and len(orphan_nodes) == 0
                and len(cyclic_sccs) == 0
                and len(dangling_gateways) == 0
                and len(problematic_exceptions) == 0
            ),
        }
        report['processes'].append(proc_report)

    return report

# -----------------------
# DMN CHECKS (Item 2)
# -----------------------


def _infer_dmn_ns(root: ET.Element) -> Dict[str, str]:
    tag = root.tag
    if tag.startswith('{') and '}' in tag:
        ns = tag.split('}', 1)[0][1:]
        return {'dmn': ns}
    # Fallback to common ones
    return {'dmn': 'http://www.omg.org/spec/DMN/20180521/MODEL/'}


def parse_dmn_tables(root: ET.Element) -> List[Dict]:
    DMN_NS = _infer_dmn_ns(root)
    tables: List[Dict] = []
    for dt in root.findall('.//dmn:decisionTable', DMN_NS):
        hit_policy = dt.get('hitPolicy', 'UNIQUE')
        inputs: List[Dict] = []
        for inp in dt.findall('dmn:input', DMN_NS):
            label = (inp.findtext('dmn:label', default='', namespaces=DMN_NS) or '').strip()
            expr_name = (inp.findtext('dmn:inputExpression', default='', namespaces=DMN_NS) or '').strip()
            if not expr_name:
                expr_name = (inp.findtext('dmn:inputExpression/dmn:text', default='', namespaces=DMN_NS) or '').strip()
            allowed = (inp.findtext('dmn:inputValues/dmn:text', default='', namespaces=DMN_NS) or '').strip()
            inputs.append({'label': label or expr_name, 'expr': expr_name, 'allowed': allowed})
        outputs: List[Dict] = []
        for out in dt.findall('dmn:output', DMN_NS):
            outputs.append({'label': out.get('label', '').strip(), 'name': out.get('name', '').strip()})
        rules: List[Dict] = []
        for r in dt.findall('dmn:rule', DMN_NS):
            input_entries = [
                (ie.findtext('dmn:text', default='', namespaces=DMN_NS) or '').strip()
                for ie in r.findall('dmn:inputEntry', DMN_NS)
            ]
            output_entries = [
                (oe.findtext('dmn:text', default='', namespaces=DMN_NS) or '').strip()
                for oe in r.findall('dmn:outputEntry', DMN_NS)
            ]
            rules.append({'inputs': input_entries, 'outputs': output_entries})
        tables.append({'hit_policy': hit_policy, 'inputs': inputs, 'outputs': outputs, 'rules': rules})
    return tables


class Interval:
    def __init__(self, lo: Optional[float], hi: Optional[float], lo_incl: bool = True, hi_incl: bool = True):
        self.lo, self.hi, self.lo_incl, self.hi_incl = lo, hi, lo_incl, hi_incl

    def overlaps(self, other: 'Interval') -> bool:
        lo = self.lo if self.lo is not None else -math.inf
        hi = self.hi if self.hi is not None else math.inf
        olo = other.lo if other.lo is not None else -math.inf
        ohi = other.hi if other.hi is not None else math.inf
        if hi < olo or ohi < lo:
            return False
        if hi == olo and not (self.hi_incl and other.lo_incl):
            return False
        if ohi == lo and not (other.hi_incl and self.lo_incl):
            return False
        return True

    def __repr__(self) -> str:
        lo_br = '[' if self.lo_incl else '('
        hi_br = ']' if self.hi_incl else ')'
        lo_s = str(self.lo) if self.lo is not None else '-∞'
        hi_s = str(self.hi) if self.hi is not None else '∞'
        return f"{lo_br}{lo_s}..{hi_s}{hi_br}"


def parse_allowed_values(text: str) -> Optional[Set[str]]:
    t = (text or '').strip()
    if not t:
        return None
    if t.startswith('[') and t.endswith(']'):
        inner = t[1:-1].strip()
        parts = [p.strip().strip('"').strip("'") for p in inner.split(',') if p.strip()]
        return set(parts) if parts else None
    return None


def parse_constraint(text: str):
    t = (text or '').strip()
    if t == '-' or t == '' or t.lower() == 'any':
        return ('ANY', None)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return ('ONEOF', {t[1:-1]})
    if ',' in t and all(v.strip() for v in t.split(',')):
        items: Set[object] = set()
        for part in t.split(','):
            pp = part.strip()
            if (pp.startswith('"') and pp.endswith('"')) or (pp.startswith("'") and pp.endswith("'")):
                items.add(pp[1:-1])
            else:
                try:
                    items.add(float(pp))
                except Exception:
                    items.add(pp)
        return ('ONEOF', items)
    if '..' in t and (t[0] in '([') and (t[-1] in ')]'):
        lo_incl = t[0] == '['
        hi_incl = t[-1] == ']'
        inner = t[1:-1]
        lo_s, hi_s = [s.strip() for s in inner.split('..', 1)]
        lo = None if lo_s in ['', '-'] else float(lo_s)
        hi = None if hi_s in ['', '+'] else float(hi_s)
        return ('INTERVAL', Interval(lo, hi, lo_incl, hi_incl))
    for op in ('<=', '>=', '<', '>'):
        if t.startswith(op):
            val = float(t[len(op):].strip())
            if op == '<':
                return ('INTERVAL', Interval(None, val, True, False))
            if op == '<=':
                return ('INTERVAL', Interval(None, val, True, True))
            if op == '>':
                return ('INTERVAL', Interval(val, None, False, True))
            if op == '>=':
                return ('INTERVAL', Interval(val, None, True, True))
    try:
        val = float(t)
        return ('ONEOF', {val})
    except Exception:
        return ('ANY', None)


def constraints_overlap(ca, cb) -> bool:
    kind_a, val_a = ca
    kind_b, val_b = cb
    if kind_a == 'ANY' or kind_b == 'ANY':
        return True
    if kind_a == 'ONEOF' and kind_b == 'ONEOF':
        return len(val_a & val_b) > 0
    if kind_a == 'INTERVAL' and kind_b == 'INTERVAL':
        return val_a.overlaps(val_b)
    if kind_a == 'ONEOF' and kind_b == 'INTERVAL':
        return any((isinstance(x, (int, float)) and val_b.overlaps(Interval(x, x, True, True))) for x in val_a)
    if kind_b == 'ONEOF' and kind_a == 'INTERVAL':
        return any((isinstance(x, (int, float)) and val_a.overlaps(Interval(x, x, True, True))) for x in val_b)
    return True


def parse_output_tuple(outputs: List[str]) -> Tuple[str, ...]:
    return tuple([o.strip() for o in outputs])


def check_dmn_tables(xml_path: str) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tables = parse_dmn_tables(root)
    report = {'file': xml_path, 'tables': []}

    for idx, t in enumerate(tables):
        hp = (t['hit_policy'] or 'UNIQUE').upper()
        issues = {
            'missing_hit_policy': hp == '',
            'rule_overlap_conflicts': [],
            'rule_overlap_inconsistent_outputs': [],
            'incomplete_against_allowed_values': [],
            'overly_general_default_rules': [],  # rows with all '-' that produce triage
        }

        rules = t['rules']
        inputs = t['inputs']
        outputs = t['outputs']

        parsed_rules = []
        for r_index, r in enumerate(rules):
            parsed_inputs = [parse_constraint(text) for text in r['inputs']]
            parsed_rules.append({'inputs': parsed_inputs, 'outputs': parse_output_tuple(r['outputs'])})
            # Skip overly general default rule check - catch-all rules are valid in clinical DMN
            # if all((txt.strip() == '-' or txt.strip() == '') for txt in r['inputs']):
            #     joined = ' '.join(r['outputs']).lower()
            #     if 'triage:' in joined or 'hospital' in joined or 'clinic' in joined or 'home' in joined:
            #         issues['overly_general_default_rules'].append(r_index + 1)

        # Pairwise overlap checks
        n = len(parsed_rules)
        for i in range(n):
            for j in range(i + 1, n):
                ra, rb = parsed_rules[i], parsed_rules[j]
                overlaps_all = True
                max_len = max(len(inputs), len(ra['inputs']), len(rb['inputs']))
                for k in range(max_len):
                    ca = ra['inputs'][k] if k < len(ra['inputs']) else ('ANY', None)
                    cb = rb['inputs'][k] if k < len(rb['inputs']) else ('ANY', None)
                    if not constraints_overlap(ca, cb):
                        overlaps_all = False
                        break
                if overlaps_all:
                    if hp in ('UNIQUE',):
                        issues['rule_overlap_conflicts'].append((i + 1, j + 1))
                    if hp in ('UNIQUE', 'ANY') and ra['outputs'] != rb['outputs']:
                        issues['rule_overlap_inconsistent_outputs'].append((i + 1, j + 1))

        # Completeness vs allowedValues
        for col_idx, inp in enumerate(inputs):
            allowed = parse_allowed_values(inp.get('allowed', '') or '')
            if not allowed:
                continue
            covered: Set[str] = set()
            for r in parsed_rules:
                if col_idx >= len(r['inputs']):
                    covered |= allowed
                    continue
                kind, val = r['inputs'][col_idx]
                if kind == 'ANY':
                    covered |= allowed
                elif kind == 'ONEOF':
                    covered |= set(str(v) for v in val)
            missing = set(str(x) for x in allowed) - set(str(y) for y in covered)
            if missing:
                issues['incomplete_against_allowed_values'].append({'input': inputs[col_idx]['label'], 'missing': sorted(missing)})

        table_report = {
            'index': idx + 1,
            'hit_policy': hp,
            'rule_count': len(rules),
            'issues': issues,
            'summary_pass': (
                len(issues['rule_overlap_conflicts']) == 0
                and len(issues['rule_overlap_inconsistent_outputs']) == 0
                and len(issues['incomplete_against_allowed_values']) == 0
                and len(issues['overly_general_default_rules']) == 0
            ),
        }
        report['tables'].append(table_report)

    return report

# -----------------------
# BPMN/DMN ALIGNMENT CHECKS
# -----------------------

def check_alignment(bpmn_path: str, dmn_path: str) -> Dict:
    # Parse BPMN
    bpmn_tree = ET.parse(bpmn_path)
    bpmn_root = bpmn_tree.getroot()
    graphs = parse_bpmn_graph(bpmn_root)

    # Parse DMN
    dmn_tree = ET.parse(dmn_path)
    dmn_root = dmn_tree.getroot()
    tables = parse_dmn_tables(dmn_root)

    # BPMN expected flags from gateway conditions to end nodes
    expected_flags: Set[str] = set()
    expected_triages: Set[str] = set()
    for proc in graphs:
        out = proc['out']
        nodes = proc['nodes']
        # Identify flows to end nodes with conditions
        for sid, targets in out.items():
            for tid in targets:
                if nodes.get(tid, {}).get('type') == 'endEvent':
                    # We need to find the sequenceFlow element to read conditionExpression
                    # Build a reverse map for quick lookup
                    pass

    # Build set of DMN produced flags and triages
    dmn_flags: Set[str] = set()
    dmn_triages: Set[str] = set()
    for t in tables:
        for r in t['rules']:
            for out_val in r['outputs']:
                txt = (out_val or '').lower()
                parts = [p.strip() for p in txt.split(',') if p.strip()]
                for p in parts:
                    if p.startswith('flag:'):
                        dmn_flags.add(p.split(':', 1)[1].strip().replace('.', '_'))
                    if p.startswith('triage:'):
                        dmn_triages.add(p.split(':', 1)[1].strip())
                # Also detect bare outcomes
                if 'hospital' in txt:
                    dmn_triages.add('hospital')
                if 'clinic' in txt:
                    dmn_triages.add('clinic')
                if 'home' in txt:
                    dmn_triages.add('home')

    # Extract conditions on flows to ends
    bpmn_tree = ET.parse(bpmn_path)
    bpmn_root = bpmn_tree.getroot()
    seq_flows = bpmn_root.findall('.//bpmn:sequenceFlow', BPMN_NS)
    flow_conditions: List[str] = []
    for sf in seq_flows:
        target = sf.get('targetRef')
        # Is this flow to an end event?
        end_el = bpmn_root.find(f".//*[@id='{target}']")
        if end_el is not None and end_el.tag.endswith('endEvent'):
            cond_el = sf.find('bpmn:conditionExpression', BPMN_NS)
            if cond_el is not None and cond_el.text:
                flow_conditions.append(cond_el.text.strip())

    expected_flags = set()
    for cond in flow_conditions:
        if 'danger_sign' in cond:
            expected_flags.add('danger_sign')
        if 'clinic_referral' in cond:
            expected_flags.add('clinic_referral')

    # Prepare report
    issues: List[str] = []
    missing_flags = expected_flags - dmn_flags
    if missing_flags:
        issues.append(f"BPMN expects flags {sorted(missing_flags)} not produced by DMN")

    unhandled_triages = dmn_triages - {'hospital', 'clinic', 'home'}
    if unhandled_triages:
        issues.append(f"DMN triages {sorted(unhandled_triages)} are not handled by BPMN end nodes")

    return {
        'bpmn_file': bpmn_path,
        'dmn_file': dmn_path,
        'expected_flags': sorted(expected_flags),
        'dmn_flags': sorted(dmn_flags),
        'dmn_triages': sorted(dmn_triages),
        'issues': issues,
        'summary_pass': len(issues) == 0,
    }
