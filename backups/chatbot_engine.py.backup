"""
Chatbot Conversation Engine

Uses BPMN and DMN files to drive interactive medical consultation conversations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from .bpmn_parser import BPMNParser, ConversationFlow, FlowNode, SequenceFlow
from .dmn_parser import DMNParser, DecisionLogic, DecisionTable, DecisionRule
import re


@dataclass
class ConversationState:
    """Tracks the current state of a conversation"""
    current_node_id: str
    collected_data: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    outcome: Optional[str] = None
    flags: Dict[str, bool] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)


@dataclass
class Question:
    """Represents a question to ask the user"""
    variable: str
    text: str
    question_type: str  # 'boolean', 'numeric', 'choice'
    choices: Optional[List[str]] = None
    validation: Optional[str] = None


class ChatbotEngine:
    """Main conversation engine that uses BPMN/DMN for medical consultations"""
    
    def __init__(self, bpmn_path: str, dmn_path: str):
        """Initialize chatbot with BPMN and DMN files"""
        self.bpmn_parser = BPMNParser()
        self.dmn_parser = DMNParser()
        
        # Parse the input files
        self.conversation_flow = self.bpmn_parser.parse_bpmn_file(bpmn_path)
        self.decision_logic = self.dmn_parser.parse_dmn_file(dmn_path)
        
        # Create variable mappings for questions
        self.questions = self._create_questions()
        
        # Track required variables from DMN
        self.required_variables = self._extract_required_variables()

        # Define a minimal core assessment set to collect before evaluating DMN
        default_core = [
            'convulsion', 'edema_both_feet', 'blood_in_stool',
            'temp', 'muac_mm', 'resp_rate', 'feels_very_hot', 'diarrhea_days',
            'rdt_result', 'unusually_sleepy', 'not_able_to_drink', 'vomits_everything',
            'chest_indrawing'
        ]
        # Only keep those present in the DMN variables/questions
        self.core_variables = [v for v in default_core if v in self.questions]
    
    def start_conversation(self) -> ConversationState:
        """Start a new conversation"""
        return ConversationState(
            current_node_id=self.conversation_flow.start_node,
            collected_data={},
            conversation_history=[],
            outcome=None,
            flags={},
            reasoning=[]
        )
    
    def get_next_question(self, state: ConversationState) -> Optional[Question]:
        """Get the next question to ask based on current state"""
        # Always prioritize collecting core assessment variables first
        missing_core = [v for v in self.core_variables if v not in state.collected_data]
        if missing_core:
            var_name = missing_core[0]
            return self.questions.get(var_name, self._create_default_question(var_name))

        # Then check if we need any DMN variables for decision making
        missing_vars = self._find_missing_variables(state.collected_data)
        
        if missing_vars:
            # Ask for the first missing variable
            var_name = missing_vars[0]
            if var_name in self.questions:
                return self.questions[var_name]
            else:
                # Create a default question for unknown variables
                return self._create_default_question(var_name)
        
        # If no missing variables from DMN, check for any missing question variables
        # This makes the system data-driven based on available questions
        for var_name in self.questions:
            if var_name not in state.collected_data:
                return self.questions[var_name]
        
        return None
    
    def process_answer(self, state: ConversationState, variable: str, answer: Any) -> tuple[ConversationState, bool]:
        """Process user's answer and update conversation state. Returns (state, is_valid)"""
        # Validate the answer
        validated_answer, is_valid = self._validate_answer(variable, answer)
        
        if not is_valid:
            return state, False  # Don't update state if invalid
        
        # Store the validated answer
        state.collected_data[variable] = validated_answer
        
        # Add to conversation history
        question_text = self.questions.get(variable, Question(variable, f"Value for {variable}", "string")).text
        state.conversation_history.append({
            'question': question_text,
            'answer': str(validated_answer)
        })
        
        # Try to make decisions with current data
        self._evaluate_decisions(state)
        
        # Check if we can progress in the flow
        self._progress_conversation_flow(state)
        
        return state, True
    
    def is_conversation_complete(self, state: ConversationState) -> bool:
        """Check if the conversation has reached a conclusion"""
        return state.outcome is not None
    
    def get_final_recommendation(self, state: ConversationState) -> Dict[str, Any]:
        """Get the final recommendation and reasoning"""
        return {
            'outcome': state.outcome,
            'reasoning': state.reasoning,
            'collected_data': state.collected_data,
            'flags': state.flags,
            'conversation_history': state.conversation_history
        }
    
    def _create_questions(self) -> Dict[str, Question]:
        """Create user-friendly questions for variables found in DMN"""
        questions = {}
        
        # Standard medical questions
        standard_questions = {
            'temp': Question('temp', 'What is the patient\'s temperature in °C?', 'numeric'),
            'resp_rate': Question('resp_rate', 'What is the patient\'s respiratory rate (breaths per minute)?', 'numeric'),
            'muac_mm': Question('muac_mm', 'What is the MUAC measurement in millimeters?', 'numeric'),
            'age_months': Question('age_months', 'How old is the patient in months?', 'numeric'),
            'convulsion': Question('convulsion', 'Has the patient had convulsions or fits?', 'boolean'),
            'feels_very_hot': Question('feels_very_hot', 'Does the patient feel very hot to touch?', 'boolean'),
            'blood_in_stool': Question('blood_in_stool', 'Is there blood in the stool?', 'boolean'),
            'diarrhea_days': Question('diarrhea_days', 'How many days has the patient had diarrhea?', 'numeric'),
            'edema_both_feet': Question('edema_both_feet', 'Does the patient have swelling (edema) in both feet?', 'boolean'),
            'malaria_present': Question('malaria_present', 'Is malaria known to be present in this area?', 'boolean'),
            'cholera_present': Question('cholera_present', 'Is cholera known to be present in this area?', 'boolean'),
            'sex': Question('sex', 'What is the patient\'s sex?', 'choice', choices=['m', 'f'])
        }
        
        # Add any additional variables found in the DMN
        for var_name, var_type in self.decision_logic.variable_types.items():
            if var_name not in standard_questions:
                questions[var_name] = self._create_default_question(var_name, var_type)
            else:
                questions[var_name] = standard_questions[var_name]
        
        # Add standard questions that might be referenced
        for var_name, question in standard_questions.items():
            if var_name not in questions:
                questions[var_name] = question
        
        return questions
    
    def _create_default_question(self, var_name: str, var_type: str = 'string') -> Question:
        """Create a default question for an unknown variable"""
        # Convert variable name to human-readable text
        readable_name = var_name.replace('_', ' ').title()
        
        if var_type == 'boolean':
            return Question(var_name, f'Is the patient showing signs of {readable_name.lower()}?', 'boolean')
        elif var_type == 'numeric':
            return Question(var_name, f'What is the {readable_name.lower()} value?', 'numeric')
        else:
            return Question(var_name, f'Please describe {readable_name.lower()}:', 'string')
    
    def _extract_required_variables(self) -> List[str]:
        """Extract all variables that might be needed for decisions"""
        required_vars = set()
        
        # Get variables from DMN input data
        for var_name in self.decision_logic.input_data.values():
            required_vars.add(var_name)
        
        # Get variables from decision table input expressions
        for decision in self.decision_logic.decisions.values():
            for input_col in decision.input_columns.values():
                if input_col.input_expression:
                    required_vars.add(input_col.input_expression)
        
        return list(required_vars)
    
    def _find_missing_variables(self, collected_data: Dict[str, Any]) -> List[str]:
        """Find variables that haven't been collected yet"""
        missing = []
        
        # Check which variables are needed for current decisions
        for decision in self.decision_logic.decisions.values():
            for input_col in decision.input_columns.values():
                var_name = input_col.input_expression
                if var_name and var_name not in collected_data:
                    if var_name not in missing:
                        missing.append(var_name)
        
        return missing
    
    def _validate_answer(self, variable: str, answer: Any) -> tuple[Any, bool]:
        """Validate and convert answer to appropriate type. Returns (value, is_valid)"""
        if variable not in self.questions:
            return answer, True
        
        question = self.questions[variable]
        
        if question.question_type == 'boolean':
            if isinstance(answer, bool):
                return answer, True
            elif isinstance(answer, str):
                answer_lower = answer.lower().strip()
                if answer_lower in ['yes', 'y', 'true']:
                    return True, True
                elif answer_lower in ['no', 'n', 'false']:
                    return False, True
                else:
                    return None, False  # Invalid boolean input
            else:
                return None, False
        
        elif question.question_type == 'numeric':
            try:
                value = float(answer)
                return value, True
            except:
                return None, False  # Invalid numeric input
        
        elif question.question_type == 'choice':
            if question.choices:
                answer_lower = str(answer).lower().strip()
                for choice in question.choices:
                    if answer_lower == choice.lower():
                        return choice, True
                return None, False  # Invalid choice
            else:
                return str(answer), True
        
        else:
            return str(answer), True
    
    def _evaluate_decisions(self, state: ConversationState):
        """Evaluate DMN decision tables with current data"""
        # Clear previous reasoning
        state.reasoning.clear()
        state.flags.clear()
        
        # Only evaluate decisions if we have enough data
        if not state.collected_data:
            return

        # Guard: require core assessment variables before evaluating
        if any(v not in state.collected_data for v in self.core_variables):
            return
            
        # Evaluate each decision table
        for decision_id, decision_table in self.decision_logic.decisions.items():
            # Only evaluate if we have the required input data for this decision
            required_inputs = set()
            for input_col in decision_table.input_columns.values():
                if input_col.input_expression:
                    required_inputs.add(input_col.input_expression)
            
            # Check if we have at least some of the required inputs
            available_inputs = set(state.collected_data.keys())
            if not required_inputs.intersection(available_inputs):
                continue  # Skip this decision if no relevant data
            
            matching_rules = self.dmn_parser.find_matching_rules(decision_table, state.collected_data)
            
            # Filter to only the most specific matching rules (avoid catch-all rules)
            # For FIRST hit policy, just take the first matching rule to respect priority order
            if decision_table.hit_policy == 'FIRST' and matching_rules:
                specific_rules = [matching_rules[0]]
            else:
                specific_rules = self._filter_most_specific_rules(matching_rules, decision_table)
            
            for rule in specific_rules:
                # Process rule outputs
                for output_id, value in rule.output_entries.items():
                    if value and value.strip():
                        # Parse comma-separated effects (e.g., "triage:clinic, reason:muac.low")
                        effects = [effect.strip() for effect in value.split(',')]
                        
                        for effect in effects:
                            if ':' in effect:
                                effect_type, effect_value = effect.split(':', 1)
                                effect_type = effect_type.strip()
                                effect_value = effect_value.strip()
                                
                                if effect_type == 'triage':
                                    if effect_value.lower() in ['hospital', 'clinic', 'home']:
                                        new_outcome = effect_value.lower().title()
                                        
                                        # Hospital overrides Clinic overrides Home
                                        if (state.outcome is None or 
                                            (new_outcome == 'Hospital') or 
                                            (new_outcome == 'Clinic' and state.outcome == 'Home')):
                                            state.outcome = new_outcome
                                            state.reasoning.append(f"Recommendation: {state.outcome}")
                                        
                                        # Set BPMN-compatible flags for gateway conditions
                                        if effect_value.lower() == 'clinic':
                                            state.flags['clinic_referral'] = True
                                        elif effect_value.lower() == 'hospital':
                                            state.flags['danger_sign'] = True
                                
                                elif effect_type == 'flag':
                                    # Normalize flag names for BPMN compatibility
                                    flag_name = effect_value.replace('.', '_')  # danger.sign -> danger_sign
                                    state.flags[flag_name] = True
                                    state.reasoning.append(f"Clinical flag: {flag_name}")
                                
                                elif effect_type == 'reason':
                                    state.reasoning.append(f"Clinical finding: {effect_value}")
                            
                            else:
                                # Handle single values without type prefix
                                if effect.lower() in ['hospital', 'clinic', 'home']:
                                    state.outcome = effect.lower().title()
                                    state.reasoning.append(f"Recommendation: {state.outcome}")
                                else:
                                    state.reasoning.append(f"Finding: {effect}")
    
    def _progress_conversation_flow(self, state: ConversationState):
        """Progress through the BPMN conversation flow based on current state"""
        current_node = self.conversation_flow.nodes.get(state.current_node_id)
        if not current_node:
            return
        
        # If we're at a decision gateway, try to follow conditional flows
        if current_node.node_type == 'exclusiveGateway':
            next_node = self._evaluate_gateway_conditions(state, current_node)
            if next_node:
                state.current_node_id = next_node
                
                # Check if we've reached an end node
                if next_node in self.conversation_flow.end_nodes:
                    state.outcome = self.conversation_flow.end_nodes[next_node]
        
        # If we're at a task, move to the next node
        elif current_node.node_type == 'task':
            next_nodes = self.bpmn_parser.get_next_nodes(state.current_node_id, self.conversation_flow.flows)
            if next_nodes:
                state.current_node_id = next_nodes[0]
    
    def _evaluate_gateway_conditions(self, state: ConversationState, gateway_node: FlowNode) -> Optional[str]:
        """Evaluate gateway conditions to determine next node"""
        outgoing_flows = self.bpmn_parser.get_outgoing_flows(gateway_node.id, self.conversation_flow.flows)
        
        for flow in outgoing_flows:
            if self._evaluate_flow_condition(flow, state):
                return flow.target_ref
        
        return None
    
    def _evaluate_flow_condition(self, flow: SequenceFlow, state: ConversationState) -> bool:
        """Evaluate a sequence flow condition"""
        if not flow.condition:
            return True
        
        condition = flow.condition.strip()
        
        # Handle common condition patterns
        if condition == 'else' or condition == 'default':
            return True
        
        # Evaluate flag conditions
        if '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected_value = parts[1].strip().strip('"\'')
                
                # Check flags
                if var_name in state.flags:
                    return str(state.flags[var_name]).lower() == expected_value.lower()
                
                # Check collected data
                if var_name in state.collected_data:
                    return str(state.collected_data[var_name]).lower() == expected_value.lower()
        
        return False
    
    def _filter_most_specific_rules(self, matching_rules: List, decision_table) -> List:
        """Filter to only the most specific matching rules (avoid generic catch-all rules)"""
        if not matching_rules:
            return []
        
        # Score rules by specificity (number of non-dash conditions)
        scored_rules = []
        for rule in matching_rules:
            specificity_score = 0
            for input_id, condition in rule.input_entries.items():
                if condition and condition.strip() != '-':
                    specificity_score += 1
            scored_rules.append((rule, specificity_score))
        
        # Return only rules with the highest specificity score
        if scored_rules:
            max_score = max(score for _, score in scored_rules)
            # Only return rules that have at least one specific condition
            specific_rules = [rule for rule, score in scored_rules if score == max_score and score > 0]
            
            # If no specific rules, return the first rule as fallback
            if not specific_rules and scored_rules:
                return [scored_rules[0][0]]
            
            return specific_rules
        
        return []
