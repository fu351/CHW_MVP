"""
Chatbot Conversation Engine - Redesigned for Sequential Clinical Workflow

Uses redesigned BPMN and DMN files to drive proper sequential medical consultations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from .bpmn_parser import BPMNParser, ConversationFlow, FlowNode, SequenceFlow
from .dmn_parser import DMNParser, DecisionLogic, DecisionTable, DecisionRule
import re


@dataclass
class ConversationState:
    """Tracks the current state of a conversation"""
    current_step: str  # 'danger_signs', 'clinical_assessment', 'complete'
    collected_data: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    outcome: Optional[str] = None
    flags: Dict[str, bool] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    danger_signs_assessed: bool = False
    clinical_assessment_complete: bool = False


@dataclass
class Question:
    """Represents a question to ask the user"""
    variable: str
    text: str
    question_type: str  # 'boolean', 'numeric', 'choice'
    choices: Optional[List[str]] = None
    validation: Optional[str] = None


class ChatbotEngine:
    """Main conversation engine for sequential clinical assessment"""
    
    def __init__(self, bpmn_path: str, dmn_path: str):
        """Initialize chatbot with redesigned BPMN and DMN files"""
        self.bpmn_parser = BPMNParser()
        self.dmn_parser = DMNParser()
        
        # Parse the input files
        self.conversation_flow = self.bpmn_parser.parse_bpmn_file(bpmn_path)
        self.decision_logic = self.dmn_parser.parse_dmn_file(dmn_path)
        
        # Create variable mappings for questions
        self.questions = self._create_questions()
        
        # Define clinical workflow steps
        self.danger_sign_variables = self._extract_danger_sign_variables()
        self.clinical_variables = self._extract_clinical_variables()
    
    def start_conversation(self) -> ConversationState:
        """Start a new conversation with danger signs assessment"""
        return ConversationState(
            current_step="danger_signs",
            collected_data={},
            conversation_history=[],
            outcome=None,
            flags={},
            reasoning=[],
            danger_signs_assessed=False,
            clinical_assessment_complete=False
        )
    
    def get_next_question(self, state: ConversationState) -> Optional[Question]:
        """Get the next question based on current workflow step"""
        
        if state.current_step == "danger_signs":
            # First assess danger signs
            missing_danger = [v for v in self.danger_sign_variables if v not in state.collected_data]
            if missing_danger:
                var_name = missing_danger[0]
                return self.questions.get(var_name, self._create_default_question(var_name))
            
            # If all danger signs collected, evaluate them
            if not state.danger_signs_assessed:
                self._evaluate_danger_signs(state)
                state.danger_signs_assessed = True
                
                # If hospital referral, we're done
                if state.outcome == "Hospital":
                    state.current_step = "complete"
                    return None
                
                # Otherwise, move to clinical assessment
                state.current_step = "clinical_assessment"
                return self.get_next_question(state)  # Recursive call for next step
        
        elif state.current_step == "clinical_assessment":
            # Collect remaining clinical variables
            missing_clinical = [v for v in self.clinical_variables if v not in state.collected_data]
            if missing_clinical:
                var_name = missing_clinical[0]
                return self.questions.get(var_name, self._create_default_question(var_name))
            
            # If all clinical data collected, make final assessment
            if not state.clinical_assessment_complete:
                self._evaluate_clinical_assessment(state)
                state.clinical_assessment_complete = True
                state.current_step = "complete"
                return None
        
        return None  # Conversation complete
    
    def process_answer(self, state: ConversationState, variable: str, answer: Any) -> tuple[ConversationState, bool]:
        """Process user's answer and update conversation state"""
        # Validate the answer
        validated_answer, is_valid = self._validate_answer(variable, answer)
        
        if not is_valid:
            return state, False
        
        # Store the validated answer
        state.collected_data[variable] = validated_answer
        
        # Add to conversation history
        question_text = self.questions.get(variable, Question(variable, f"Value for {variable}", "string")).text
        state.conversation_history.append({
            'question': question_text,
            'answer': str(validated_answer)
        })
        
        return state, True
    
    def is_conversation_complete(self, state: ConversationState) -> bool:
        """Check if the conversation has reached a conclusion"""
        return state.current_step == "complete" or state.outcome is not None
    
    def get_final_recommendation(self, state: ConversationState) -> Dict[str, Any]:
        """Get the final recommendation and reasoning"""
        return {
            'outcome': state.outcome or "Home",
            'reasoning': state.reasoning,
            'collected_data': state.collected_data,
            'flags': state.flags,
            'conversation_history': state.conversation_history
        }
    
    def _extract_danger_sign_variables(self) -> List[str]:
        """Extract variables needed for danger signs assessment"""
        danger_variables = []
        
        # Look for danger signs decision in DMN
        if 'danger_signs_decision' in self.decision_logic.decisions:
            decision = self.decision_logic.decisions['danger_signs_decision']
            for input_col in decision.input_columns.values():
                if input_col.input_expression:
                    danger_variables.append(input_col.input_expression)
        
        # Default danger sign variables if DMN parsing fails
        if not danger_variables:
            danger_variables = [
                'convulsion', 'unconscious', 'vomits_everything', 
                'cannot_drink_or_feed', 'chest_indrawing'
            ]
        
        return danger_variables
    
    def _extract_clinical_variables(self) -> List[str]:
        """Extract variables needed for clinical assessment"""
        clinical_variables = []
        
        # Look for clinical assessment decision in DMN
        if 'clinical_assessment' in self.decision_logic.decisions:
            decision = self.decision_logic.decisions['clinical_assessment']
            for input_col in decision.input_columns.values():
                if input_col.input_expression and input_col.input_expression not in self.danger_sign_variables:
                    clinical_variables.append(input_col.input_expression)
        
        # Default clinical variables if DMN parsing fails
        if not clinical_variables:
            clinical_variables = [
                'temp', 'resp_rate', 'muac_mm', 'age_months',
                'diarrhea', 'fever', 'cough', 'blood_in_stool'
            ]
        
        return clinical_variables
    
    def _evaluate_danger_signs(self, state: ConversationState):
        """Evaluate danger signs and determine if immediate hospital referral needed"""
        if 'danger_signs_decision' not in self.decision_logic.decisions:
            return
        
        decision_table = self.decision_logic.decisions['danger_signs_decision']
        matching_rules = self.dmn_parser.find_matching_rules(decision_table, state.collected_data)
        
        for rule in matching_rules:
            for output_id, value in rule.output_entries.items():
                if value and value.strip().lower() == "true":
                    # Danger sign detected
                    state.outcome = "Hospital"
                    state.flags['danger_sign'] = True
                    state.reasoning.append("Danger signs detected - immediate hospital referral required")
                    return
    
    def _evaluate_clinical_assessment(self, state: ConversationState):
        """Evaluate clinical assessment and make final triage decision"""
        if 'clinical_assessment' not in self.decision_logic.decisions:
            state.outcome = "Home"
            return
        
        decision_table = self.decision_logic.decisions['clinical_assessment']
        matching_rules = self.dmn_parser.find_matching_rules(decision_table, state.collected_data)
        
        # Use first matching rule (FIRST hit policy)
        if matching_rules:
            rule = matching_rules[0]
            for output_id, value in rule.output_entries.items():
                if value and value.strip():
                    triage_decision = value.strip().lower()
                    if triage_decision in ['hospital', 'clinic', 'home']:
                        state.outcome = triage_decision.title()
                        state.reasoning.append(f"Clinical assessment recommends: {state.outcome}")
                        
                        # Set appropriate flags
                        if triage_decision == 'clinic':
                            state.flags['clinic_referral'] = True
                        elif triage_decision == 'hospital':
                            state.flags['danger_sign'] = True
                        
                        return
        
        # Default to home care if no specific recommendation
        state.outcome = "Home"
        state.reasoning.append("No specific concerns - home care recommended")
    
    def _create_questions(self) -> Dict[str, Question]:
        """Create user-friendly questions for variables"""
        questions = {}
        
        # Standard medical questions with clinical context
        standard_questions = {
            # Danger signs
            'convulsion': Question('convulsion', '🚨 Has the child had convulsions (fits) recently?', 'boolean'),
            'unconscious': Question('unconscious', '🚨 Is the child unconscious or unresponsive?', 'boolean'),
            'vomits_everything': Question('vomits_everything', '🚨 Does the child vomit everything they eat or drink?', 'boolean'),
            'cannot_drink_or_feed': Question('cannot_drink_or_feed', '🚨 Is the child unable to drink or breastfeed?', 'boolean'),
            'chest_indrawing': Question('chest_indrawing', '🚨 Does the child have severe chest indrawing (pulling in below the ribs)?', 'boolean'),
            
            # Clinical assessment
            'temp': Question('temp', '🌡️ What is the child\'s temperature in °C? (Enter number like 38.5)', 'numeric'),
            'resp_rate': Question('resp_rate', '💨 Count breaths for one minute. How many breaths per minute?', 'numeric'),
            'muac_mm': Question('muac_mm', '📏 Measure upper arm circumference. What is the MUAC in mm?', 'numeric'),
            'age_months': Question('age_months', '👶 How old is the child in months?', 'numeric'),
            'diarrhea': Question('diarrhea', '💧 Does the child have diarrhea (loose, watery stools)?', 'boolean'),
            'fever': Question('fever', '🔥 Does the child have fever or feel hot to touch?', 'boolean'),
            'cough': Question('cough', '😷 Does the child have a cough?', 'boolean'),
            'blood_in_stool': Question('blood_in_stool', '🩸 Is there blood in the child\'s stool?', 'boolean'),
            'edema_both_feet': Question('edema_both_feet', '🦶 Does the child have swelling in both feet?', 'boolean'),
            'unusually_sleepy': Question('unusually_sleepy', '😴 Is the child unusually sleepy or lethargic?', 'boolean')
        }
        
        # Add any additional variables from DMN
        for var_name, var_type in self.decision_logic.variable_types.items():
            if var_name not in standard_questions:
                questions[var_name] = self._create_default_question(var_name, var_type)
            else:
                questions[var_name] = standard_questions[var_name]
        
        # Ensure all standard questions are included
        for var_name, question in standard_questions.items():
            if var_name not in questions:
                questions[var_name] = question
        
        return questions
    
    def _create_default_question(self, var_name: str, var_type: str = 'string') -> Question:
        """Create a default question for an unknown variable"""
        readable_name = var_name.replace('_', ' ').title()
        
        if var_type == 'boolean':
            return Question(var_name, f'Does the patient show signs of {readable_name.lower()}?', 'boolean')
        elif var_type == 'numeric':
            return Question(var_name, f'What is the {readable_name.lower()} value?', 'numeric')
        else:
            return Question(var_name, f'Please describe {readable_name.lower()}:', 'string')
    
    def _validate_answer(self, variable: str, answer: Any) -> tuple[Any, bool]:
        """Validate and convert answer to appropriate type"""
        if variable not in self.questions:
            return answer, True
        
        question = self.questions[variable]
        
        if question.question_type == 'boolean':
            if isinstance(answer, bool):
                return answer, True
            elif isinstance(answer, str):
                answer_lower = answer.lower().strip()
                if answer_lower in ['yes', 'y', 'true', '1']:
                    return True, True
                elif answer_lower in ['no', 'n', 'false', '0']:
                    return False, True
                else:
                    return None, False
            else:
                return None, False
        
        elif question.question_type == 'numeric':
            try:
                value = float(answer)
                return value, True
            except:
                return None, False
        
        elif question.question_type == 'choice':
            if question.choices:
                answer_str = str(answer).lower().strip()
                for choice in question.choices:
                    if answer_str == choice.lower():
                        return choice, True
                return None, False
            else:
                return str(answer), True
        
        else:
            return str(answer), True