"""
Dynamic BPMN-driven chatbot engine that follows workflow structure instead of hardcoded questions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import xml.etree.ElementTree as ET
from .bpmn_parser import BPMNParser
from .dmn_parser import DMNParser
from .engine import decide


@dataclass
class WorkflowState:
    """State of the workflow execution"""
    current_node_id: str
    collected_data: Dict[str, Any] = field(default_factory=dict)
    process_variables: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    completed_tasks: Set[str] = field(default_factory=set)
    outcome: Optional[str] = None
    reasoning: List[str] = field(default_factory=list)


@dataclass
class TaskQuestion:
    """Question generated from a BPMN task"""
    task_id: str
    task_name: str
    questions: List[Dict[str, str]]  # List of questions for this task
    required_variables: List[str]


class DynamicChatbotEngine:
    """BPMN-driven chatbot engine that dynamically generates questions based on workflow."""
    
    def __init__(self, bpmn_file: str, dmn_file: str):
        """Initialize with BPMN workflow and DMN decision logic."""
        self.bpmn_parser = BPMNParser()
        self.dmn_parser = DMNParser()
        self.dmn_file = dmn_file
        
        # Extract workflow structure
        self.process = self._parse_bpmn_structure(bpmn_file)
        self.start_node = self._find_start_event()
        
        # Parse DMN logic once and keep it
        try:
            self.dmn_logic = self.dmn_parser.parse_dmn_file(self.dmn_file)
        except Exception as e:
            print(f"Warning: Failed to parse DMN file: {e}")
            self.dmn_logic = None

        # Try to load ASK_PLAN next to DMN
        self.ask_plan = None
        try:
            import json, os
            dmn_dir = os.path.dirname(self.dmn_file)
            candidates = [
                os.path.join(dmn_dir, 'ask_plan.json'),
                os.path.join(dmn_dir, 'who_chw_guarded5_ask_plan.json'),
            ]
            for path in candidates:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        self.ask_plan = json.load(f)
                        break
        except Exception:
            self.ask_plan = None

        # Create question mappings from DMN variables
        self.variable_questions = self._create_variable_questions()
        
    def _parse_bpmn_structure(self, bpmn_file: str) -> Dict[str, Any]:
        """Parse BPMN XML to extract process structure."""
        tree = ET.parse(bpmn_file)
        root = tree.getroot()
        
        # Handle namespaces
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
              'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
        
        process = {}
        
        # Extract all nodes
        process['nodes'] = {}
        process['flows'] = {}
        process['gateways'] = {}
        
        # Parse all elements
        for elem in root.findall('.//bpmn:*', ns):
            elem_id = elem.get('id')
            elem_name = elem.get('name', '')
            elem_type = elem.tag.split('}')[-1]  # Remove namespace
            
            if elem_type in ['startEvent', 'endEvent', 'userTask', 'businessRuleTask', 'exclusiveGateway']:
                process['nodes'][elem_id] = {
                    'id': elem_id,
                    'name': elem_name,
                    'type': elem_type
                }
                
                # Special handling for business rule tasks
                if elem_type == 'businessRuleTask':
                    decision_ref = elem.get('decisionRef')
                    if decision_ref:
                        process['nodes'][elem_id]['decisionRef'] = decision_ref
                        
            elif elem_type == 'sequenceFlow':
                source = elem.get('sourceRef')
                target = elem.get('targetRef')
                
                # Extract condition if present
                condition_elem = elem.find('.//bpmn:conditionExpression', ns)
                condition = None
                if condition_elem is not None:
                    condition = condition_elem.text
                
                process['flows'][elem_id] = {
                    'id': elem_id,
                    'source': source,
                    'target': target,
                    'condition': condition
                }
        
        return process
    
    def _find_start_event(self) -> str:
        """Find the start event node ID."""
        for node_id, node in self.process['nodes'].items():
            if node['type'] == 'startEvent':
                return node_id
        raise ValueError("No start event found in BPMN process")
    
    def _create_variable_questions(self) -> Dict[str, Dict[str, str]]:
        """Create question mappings for DMN variables."""
        questions = {}
        
        # Default question templates based on variable names
        question_templates = {
            # Danger signs
            'convulsions': {
                'text': '🚨 Has the child had convulsions (fits) recently?',
                'type': 'boolean',
                'help': 'Look for seizures, fits, or spasms'
            },
            'unconscious': {
                'text': '🚨 Is the child unconscious or unresponsive?',
                'type': 'boolean',
                'help': 'Check if child responds to voice or touch'
            },
            'unable_to_drink': {
                'text': '🚨 Is the child unable to drink or breastfeed?',
                'type': 'boolean',
                'help': 'Cannot take fluids at all'
            },
            'chest_indrawing': {
                'text': '🚨 Does the child have severe chest indrawing?',
                'type': 'boolean',
                'help': 'Lower chest wall pulls in when breathing'
            },
            'vomiting_everything': {
                'text': '🚨 Does the child vomit everything they eat or drink?',
                'type': 'boolean',
                'help': 'Cannot keep anything down'
            },
            
            # Symptoms
            'diarrhea': {
                'text': '💧 Does the child have diarrhea (loose, watery stools)?',
                'type': 'boolean',
                'help': '3 or more loose stools in 24 hours'
            },
            'diarrhoea': {
                'text': '💧 Does the child have diarrhea (loose, watery stools)?',
                'type': 'boolean',
                'help': '3 or more loose stools in 24 hours'
            },
            'fever': {
                'text': '🔥 Does the child have fever or feel hot to touch?',
                'type': 'boolean',
                'help': 'Hot to touch or measured temperature ≥37.5°C'
            },
            'cough': {
                'text': '😷 Does the child have a cough?',
                'type': 'boolean',
                'help': 'Any cough, wet or dry'
            },
            'blood_in_stool': {
                'text': '🩸 Is there blood in the child\'s stool?',
                'type': 'boolean',
                'help': 'Visible blood in diarrhea or stool'
            },
            
            # Measurements
            'muac_mm': {
                'text': '📏 Measure upper arm circumference. What is the MUAC in mm?',
                'type': 'numeric',
                'help': 'Mid-upper arm circumference in millimeters (normal >125mm)'
            },
            'resp_rate': {
                'text': '💨 Count breaths for one minute. How many breaths per minute?',
                'type': 'numeric',
                'help': 'Count chest rises for 60 seconds'
            },
            'temperature': {
                'text': '🌡️ What is the child\'s temperature in °C?',
                'type': 'numeric',
                'help': 'Measured temperature (normal <37.5°C)'
            },
            'age_months': {
                'text': '👶 How old is the child in months?',
                'type': 'numeric',
                'help': 'Age in complete months'
            },
            
            # Duration questions
            'diarrhea_duration_days': {
                'text': '📅 How many days has the child had diarrhea?',
                'type': 'numeric',
                'help': 'Number of days with loose stools'
            },
            'fever_duration_days': {
                'text': '📅 How many days has the child had fever?',
                'type': 'numeric',
                'help': 'Number of days with fever'
            },
            'cough_duration_days': {
                'text': '📅 How many days has the child had cough?',
                'type': 'numeric',
                'help': 'Number of days with cough'
            },
            
            # Special process variable
            'diarrhea_present': {
                'text': '💧 Does the child have diarrhea?',
                'type': 'boolean',
                'help': 'This determines the next steps in assessment'
            }
        }
        
        # Get variables from DMN parser
        try:
            dmn_logic = self.dmn_parser.parse_dmn_file(self.dmn_file)
            # Use the first decision table if available
            decision_tables = list(dmn_logic.decisions.values())
            if decision_tables:
                decision_table = decision_tables[0]
                for input_col in decision_table.input_columns.values():
                    var_name = input_col.input_expression or input_col.label
                    var_name = (var_name or '').strip()
                    if not var_name:
                        continue
                    var_type = dmn_logic.variable_types.get(var_name, 'string')
                    
                    if var_name in question_templates:
                        questions[var_name] = question_templates[var_name]
                    else:
                        # Generate default question
                        readable_name = var_name.replace('_', ' ').title()
                        questions[var_name] = {
                            'text': f"❓ {readable_name}?",
                            'type': 'boolean' if var_type == 'boolean' else 'numeric' if var_type in ['number', 'integer', 'numeric'] else 'text',
                            'help': f'Please provide {readable_name}'
                        }
        except Exception as e:
            print(f"Warning: Could not extract variables from DMN: {e}")
        
        # Add process-specific questions
        questions.update(question_templates)
        
        return questions
    
    def start_conversation(self) -> WorkflowState:
        """Start a new conversation at the beginning of the workflow."""
        return WorkflowState(current_node_id=self.start_node)
    
    def get_current_task_questions(self, state: WorkflowState) -> Optional[TaskQuestion]:
        """Get questions for the current task based on BPMN workflow."""
        current_node = self.process['nodes'].get(state.current_node_id)
        if not current_node:
            return None
            
        node_type = current_node['type']
        node_name = current_node['name']
        
        if node_type == 'userTask':
            # Generate questions based on task name and context
            questions = []
            required_vars = []
            
            task_name_lower = node_name.lower()
            
            if 'diarrhea' in task_name_lower or 'diarrhoea' in task_name_lower:
                if 'details' in task_name_lower:
                    # Ask details only if diarrhea is present (prefer canonical 'diarrhea')
                    has_diarrhea = state.collected_data.get('diarrhea')
                    if has_diarrhea is None:
                        # fallback to legacy names
                        has_diarrhea = state.collected_data.get('diarrhea_present') or state.collected_data.get('diarrhoea')
                    if bool(has_diarrhea) is True:
                        vars_to_ask = ['blood_in_stool', 'diarrhea_duration_days']
                    else:
                        vars_to_ask = []
                else:
                    # Initial diarrhea question - ask only the canonical variable
                    vars_to_ask = []
                    if 'diarrhea' not in state.collected_data and 'diarrhoea' not in state.collected_data:
                        vars_to_ask.append('diarrhea')
            elif 'fever' in task_name_lower or 'malaria' in task_name_lower:
                vars_to_ask = []
                # Ask primary first
                if 'fever' not in state.collected_data:
                    vars_to_ask.append('fever')
                # Follow-ups only if fever is true
                if state.collected_data.get('fever') is True:
                    if 'fever_duration_days' not in state.collected_data:
                        vars_to_ask.append('fever_duration_days')
                    if 'temperature' not in state.collected_data:
                        vars_to_ask.append('temperature')
            elif 'cough' in task_name_lower or 'pneumonia' in task_name_lower:
                vars_to_ask = []
                # Ask primary first
                if 'cough' not in state.collected_data:
                    vars_to_ask.append('cough')
                # Follow-ups only if cough is true
                if state.collected_data.get('cough') is True:
                    if 'cough_duration_days' not in state.collected_data:
                        vars_to_ask.append('cough_duration_days')
                    if 'resp_rate' not in state.collected_data:
                        vars_to_ask.append('resp_rate')
                    if 'chest_indrawing' not in state.collected_data:
                        vars_to_ask.append('chest_indrawing')
            elif 'malnutrition' in task_name_lower:
                vars_to_ask = ['muac_mm', 'age_months']
            else:
                # Check for danger signs by default
                vars_to_ask = ['convulsions', 'unconscious', 'unable_to_drink', 'vomiting_everything']

            # If ASK_PLAN is available, intersect vars_to_ask with currently relevant module plan to avoid mismatches
            if self.ask_plan:
                plan_vars = set()
                for item in self.ask_plan:
                    plan_vars.update(item.get('ask') or [])
                    for arr in (item.get('followups_if') or {}).values():
                        plan_vars.update(arr or [])
                vars_to_ask = [v for v in vars_to_ask if v in plan_vars]
            
            # Generate questions for relevant variables
            for var_name in vars_to_ask:
                if var_name in self.variable_questions:
                    question_info = self.variable_questions[var_name]
                    questions.append({
                        'variable': var_name,
                        'text': question_info['text'],
                        'type': question_info['type'],
                        'help': question_info.get('help', '')
                    })
                    required_vars.append(var_name)
            
            return TaskQuestion(
                task_id=state.current_node_id,
                task_name=node_name,
                questions=questions,
                required_variables=required_vars
            )
        
        return None
    
    def process_user_input(self, state: WorkflowState, variable: str, value: Any) -> None:
        """Process user input for a specific variable."""
        # Store the data
        state.collected_data[variable] = value
        state.process_variables[variable] = value
        
        # Sync diarrhea canonical variable across synonyms
        if variable == 'diarrhea':
            state.collected_data['diarrhoea'] = bool(value)
            state.process_variables['diarrhoea'] = bool(value)
            state.collected_data['diarrhea_present'] = bool(value)
            state.process_variables['diarrhea_present'] = bool(value)
        elif variable == 'diarrhoea':
            state.collected_data['diarrhea'] = bool(value)
            state.process_variables['diarrhea'] = bool(value)
            state.collected_data['diarrhea_present'] = bool(value)
            state.process_variables['diarrhea_present'] = bool(value)
        elif variable == 'diarrhea_present':
            state.collected_data['diarrhea'] = bool(value)
            state.process_variables['diarrhea'] = bool(value)
            state.collected_data['diarrhoea'] = bool(value)
            state.process_variables['diarrhoea'] = bool(value)
        
        # Add to conversation history
        question_info = self.variable_questions.get(variable, {})
        state.conversation_history.append({
            'variable': variable,
            'question': question_info.get('text', f'Value for {variable}'),
            'answer': str(value)
        })
    
    def advance_workflow(self, state: WorkflowState) -> bool:
        """Advance to the next node in the workflow. Returns True if conversation continues."""
        current_node = self.process['nodes'].get(state.current_node_id)
        if not current_node:
            return False
        
        node_type = current_node['type']
        
        if node_type == 'userTask':
            # Mark task as completed
            state.completed_tasks.add(state.current_node_id)
            
        elif node_type == 'businessRuleTask':
            # Execute DMN decision using parsed DMN tables
            decision_ref = current_node.get('decisionRef')
            if decision_ref:
                try:
                    # Evaluate per-module decisions opportunistically to short-circuit danger signs
                    if self.ask_plan:
                        module_order = [m.get('module') for m in self.ask_plan if isinstance(m, dict)]
                    else:
                        module_order = ['danger_signs', 'diarrhea', 'fever_malaria', 'respiratory', 'nutrition']
                    # Evaluate modules
                    for mod in module_order:
                        mod_id = f"decide_{mod}"
                        mod_out = self._evaluate_specific_decision(mod_id, state.collected_data)
                        if mod_out:
                            self._process_dmn_result(state, [mod_out])
                            if bool(mod_out.get('danger_sign')):
                                break
                    # Finally evaluate aggregator
                    result = self._evaluate_dmn(state.collected_data)
                    self._process_dmn_result(state, result)
                except Exception as e:
                    state.reasoning.append(f"Decision evaluation failed: {e}")
        
        # Evaluate DMN before routing, so flags/triage are current for gateway conditions
        try:
            result = self._evaluate_dmn(state.collected_data)
            self._process_dmn_result(state, result)
        except Exception:
            pass

        # Find next node
        next_node_id = self._find_next_node(state)
        if next_node_id:
            state.current_node_id = next_node_id
            next_node = self.process['nodes'].get(next_node_id)
            
            if next_node and next_node['type'] == 'endEvent':
                # Workflow complete
                state.outcome = next_node['name']
                return False
            
            return True
        
        return False
    
    def _find_next_node(self, state: WorkflowState) -> Optional[str]:
        """Find the next node based on current state and flow conditions."""
        current_node_id = state.current_node_id
        
        # Find outgoing flows
        outgoing_flows = []
        for flow_id, flow in self.process['flows'].items():
            if flow['source'] == current_node_id:
                outgoing_flows.append(flow)
        
        if not outgoing_flows:
            return None
        
        # Evaluate conditions
        for flow in outgoing_flows:
            condition = flow.get('condition')
            if condition:
                # Evaluate condition
                if self._evaluate_condition(condition, state):
                    return flow['target']
            else:
                # No condition - this is the default flow
                return flow['target']
        
        # If no conditions matched, take the first flow
        return outgoing_flows[0]['target']
    
    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """Evaluate a BPMN condition expression robustly.
        Supports forms like:
        - danger_sign == true | false (any spacing/case)
        - clinic_referral == true | false
        - bare identifiers like `danger_sign` meaning true
        """
        import re
        if not condition:
            return True
        text = condition.strip()
        # Support boolean comparisons and string comparisons: triage == 'hospital'|hospital
        # Accept bare flag names as truthy check
        m = re.fullmatch(r"([A-Za-z0-9_]+)\s*==\s*'?(true|false|hospital|clinic|home)'?", text, flags=re.IGNORECASE)
        if not m:
            # Try simple bare identifier (no operator)
            ident = text.strip()
            key = ident.lower()
            val = state.flags.get(key)
            if val is None:
                val = state.process_variables.get(key)
            return bool(val)
        ident = m.group(1)
        expected_raw = m.group(2)
        key = ident.lower()
        # Resolve current value from flags first, then process vars
        cur = state.flags.get(key)
        if cur is None:
            cur = state.process_variables.get(key)
        # If expected is triage string
        if str(expected_raw).lower() in ['hospital', 'clinic', 'home']:
            cur_str = str(cur).lower() if cur is not None else ''
            return cur_str == str(expected_raw).lower()
        # Else treat as boolean
        expected = True if str(expected_raw).lower() == 'true' else False
        return bool(cur) is expected
    
    def _process_dmn_result(self, state: WorkflowState, dmn_result: List[Dict[str, Any]]) -> None:
        """Process DMN decision result and extract flags."""
        if not dmn_result:
            return
        
        # Take the first result
        result = dmn_result[0]
        
        # Prefer typed outputs: triage, danger_sign, clinic_referral, reason, ref
        triage_val = result.get('triage')
        if isinstance(triage_val, str) and triage_val:
            triage_norm = str(triage_val).strip().lower()
            state.process_variables['triage'] = triage_norm
            state.reasoning.append(f"Triage recommendation: {triage_norm}")
            # Map triage levels to flags for gateway compatibility
            if triage_norm == 'hospital':
                state.flags['danger_sign'] = True
                state.process_variables['danger_sign'] = True
                state.flags['clinic_referral'] = True
                state.process_variables['clinic_referral'] = True
            elif triage_norm == 'clinic':
                state.flags['clinic_referral'] = True
                state.process_variables['clinic_referral'] = True
        # boolean flags
        for flag_key in ['danger_sign', 'clinic_referral']:
            if flag_key in result:
                try:
                    fv = bool(result.get(flag_key))
                except Exception:
                    fv = False
                if fv:
                    state.flags[flag_key] = True
                    state.process_variables[flag_key] = True
        # reason/ref as strings
        if isinstance(result.get('reason'), str) and result.get('reason'):
            state.reasoning.append(f"Clinical reason: {result.get('reason')}")
        if isinstance(result.get('ref'), str) and result.get('ref'):
            state.reasoning.append(f"Reference: {result.get('ref')}")
        
        # Backward compatibility: parse legacy 'effect' if present
        effect = result.get('effect', '')
        if isinstance(effect, str) and effect:
            parts = [part.strip() for part in effect.split(',') if part.strip()]
            import re
            def _normalize_token(s: str) -> str:
                s = s.strip().lower()
                s = re.sub(r"[^a-z0-9_]+", "_", s)
                s = re.sub(r"_+", "_", s).strip('_')
                return s
            for part in parts:
                low = part.lower()
                if low.startswith('flag:'):
                    raw_flag = part.split(':', 1)[1]
                    flag_name = _normalize_token(raw_flag)
                    state.flags[flag_name] = True
                    state.process_variables[flag_name] = True
                elif low.startswith('triage:'):
                    triage = _normalize_token(part.split(':', 1)[1])
                    state.process_variables['triage'] = triage
                    state.reasoning.append(f"Triage recommendation: {triage}")
                elif low.startswith('reason:'):
                    reason = _normalize_token(part.split(':', 1)[1]).replace('_', ' ')
                    state.reasoning.append(f"Clinical reason: {reason}")

    def _evaluate_dmn(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate the parsed DMN using current inputs and return typed outputs.
        Returns a list with a single result dict when a rule matches; otherwise empty list.
        """
        if not self.dmn_logic or not self.dmn_logic.decisions:
            return []
        # Prefer an aggregate/final decision table if present
        selected_table = None
        for table in self.dmn_logic.decisions.values():
            name_l = (table.name or '').strip().lower()
            if name_l in ['aggregate_final', 'aggregate', 'final', 'overall_recommendation']:
                selected_table = table
                break
        if selected_table is None:
            # Fallback: just use the first decision table
            selected_table = list(self.dmn_logic.decisions.values())[0]

        # Find matching rules
        matches = self.dmn_parser.find_matching_rules(selected_table, inputs)
        if not matches:
            return []

        # Respect hit policy: FIRST or UNIQUE -> take first
        match = matches[0]

        # Build output dict from output entries
        out: Dict[str, Any] = {}
        for out_id, val in match.output_entries.items():
            col = selected_table.output_columns.get(out_id)
            key = (col.name or col.label or '').strip()
            key = key.lower().replace(' ', '_') if key else f'out_{out_id.lower()}'
            parsed_val = self._parse_dmn_output_value(val)
            out[key] = parsed_val
        return [out]

    def _evaluate_specific_decision(self, decision_id: str, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.dmn_logic:
            return None
        table = None
        for did, t in self.dmn_logic.decisions.items():
            if (t.name or '').strip().lower() == decision_id.strip().lower() or did.strip().lower() == decision_id.strip().lower():
                table = t
                break
        if table is None:
            return None
        matches = self.dmn_parser.find_matching_rules(table, inputs)
        if not matches:
            return None
        m = matches[0]
        out: Dict[str, Any] = {}
        for out_id, val in m.output_entries.items():
            col = table.output_columns.get(out_id)
            key = (col.name or col.label or '').strip()
            key = key.lower().replace(' ', '_') if key else f'out_{out_id.lower()}'
            parsed_val = self._parse_dmn_output_value(val)
            out[key] = parsed_val
        return out

    @staticmethod
    def _parse_dmn_output_value(val: Any) -> Any:
        """Parse a DMN output entry textual value into Python types.
        Handles booleans and quoted strings.
        """
        if val is None:
            return None
        s = str(val).strip()
        sl = s.lower()
        if sl in ['true', 'false']:
            return sl == 'true'
        # Strip quotes if present
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s
    
    def get_final_recommendation(self, state: WorkflowState) -> Dict[str, Any]:
        """Get final recommendation with enhanced reasoning."""
        # Generate clinical reasoning based on collected data
        clinical_reasoning = self._generate_clinical_reasoning(state)
        clinical_summary = self._generate_clinical_summary(state)
        next_steps = self._generate_next_steps(state.outcome or "Unknown", state)
        
        return {
            'outcome': state.outcome,
            'reasoning': state.reasoning,
            'clinical_reasoning': clinical_reasoning,
            'clinical_summary': clinical_summary,
            'next_steps': next_steps,
            'flags': state.flags,
            'collected_data': state.collected_data,
            'conversation_history': state.conversation_history
        }
    
    def _generate_clinical_reasoning(self, state: WorkflowState) -> List[str]:
        """Generate detailed clinical reasoning."""
        reasoning = []
        data = state.collected_data
        
        # Age assessment
        age_months = data.get('age_months', 0)
        if age_months:
            if age_months < 2:
                reasoning.append(f"Very young infant ({age_months} months) - high risk category")
            elif age_months < 12:
                reasoning.append(f"Infant ({age_months} months) - requires careful assessment")
            else:
                reasoning.append(f"Child ({age_months} months old)")
        
        # Nutrition assessment
        muac = data.get('muac_mm', 0)
        if muac and muac < 115:
            reasoning.append(f"SEVERE malnutrition detected (MUAC: {muac}mm < 115mm)")
        elif muac and muac < 125:
            reasoning.append(f"Moderate malnutrition detected (MUAC: {muac}mm < 125mm)")
        elif muac:
            reasoning.append(f"Normal nutritional status (MUAC: {muac}mm ≥ 125mm)")
        
        # Fever assessment
        if data.get('fever'):
            temp = data.get('temperature', 0)
            duration = data.get('fever_duration_days', 0)
            if temp >= 39:
                reasoning.append(f"High fever ({temp}°C for {duration} days) - concerning")
            elif temp >= 37.5:
                reasoning.append(f"Fever present ({temp}°C for {duration} days)")
                
        # Respiratory assessment
        resp_rate = data.get('resp_rate', 0)
        if resp_rate and age_months:
            if age_months < 12 and resp_rate >= 50:
                reasoning.append(f"Fast breathing in infant (≥50/min): {resp_rate}/min")
            elif age_months >= 12 and resp_rate >= 40:
                reasoning.append(f"Fast breathing in child (≥40/min): {resp_rate}/min")
                
        # Danger signs
        danger_signs = []
        for sign in ['convulsions', 'unconscious', 'unable_to_drink', 'chest_indrawing', 'vomiting_everything']:
            if data.get(sign):
                danger_signs.append(sign.replace('_', ' '))
        
        if danger_signs:
            reasoning.append(f"DANGER SIGNS present: {', '.join(danger_signs)}")
            
        # Diarrhea assessment
        if data.get('diarrhoea'):
            duration = data.get('diarrhea_duration_days', 0)
            blood = data.get('blood_in_stool', False)
            if blood:
                reasoning.append(f"Bloody diarrhea for {duration} days - dysentery suspected")
            elif duration >= 14:
                reasoning.append(f"Persistent diarrhea ({duration} days)")
            else:
                reasoning.append(f"Acute diarrhea ({duration} days)")
        
        return reasoning
    
    def _generate_clinical_summary(self, state: WorkflowState) -> str:
        """Generate clinical summary."""
        data = state.collected_data
        age_months = data.get('age_months', 0)
        
        # Primary presentation
        symptoms = []
        if data.get('diarrhoea'):
            symptoms.append("diarrhea")
        if data.get('fever'):
            symptoms.append("fever")
        if data.get('cough'):
            symptoms.append("cough")
            
        if symptoms:
            return f"{age_months}-month-old child presenting with {', '.join(symptoms)}"
        else:
            return f"{age_months}-month-old child for routine assessment"
    
    def _generate_next_steps(self, outcome: str, state: WorkflowState) -> List[str]:
        """Generate actionable next steps."""
        steps = []
        
        if outcome == "Hospital":
            steps.extend([
                "🚨 URGENT: Transport to hospital immediately",
                "📞 Call emergency services if available",
                "💧 Give small, frequent sips of water during transport",
                "🌡️ Monitor vital signs continuously",
                "📋 Bring this assessment to hospital staff"
            ])
        elif outcome == "Clinic":
            steps.extend([
                "🏥 Refer to health center/clinic today",
                "📋 Explain urgent referral to family",
                "💊 Begin recommended treatment if available",
                "📞 Follow up within 24 hours",
                "⚠️ Return immediately if condition worsens"
            ])
        elif outcome == "Home":
            steps.extend([
                "🏠 Home care with close monitoring",
                "💧 Ensure adequate fluid intake",
                "🍼 Continue breastfeeding if applicable",
                "🌡️ Monitor temperature regularly",
                "📅 Return in 2 days or if condition worsens",
                "⚠️ Watch for danger signs: convulsions, unable to drink, lethargy"
            ])
        else:
            steps.append("Assessment incomplete - continue evaluation")
            
        return steps
    
    def is_conversation_complete(self, state: WorkflowState) -> bool:
        """Check if the conversation is complete."""
        return state.outcome is not None
