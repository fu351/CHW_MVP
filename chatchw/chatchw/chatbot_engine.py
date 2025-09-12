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
            # Check if we should stop early due to critical findings
            if self._should_refer_to_hospital(state.collected_data):
                state.outcome = "Hospital"
                state.current_step = "complete"
                return None
            
            # Intelligent clinical assessment with branching logic
            next_var = self._get_next_clinical_variable(state)
            if next_var:
                return self.questions.get(next_var, self._create_default_question(next_var))
            
            # If all relevant clinical data collected, make final assessment
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
        """Get the final recommendation with detailed clinical reasoning and next steps"""
        outcome = state.outcome or "Home"
        
        # Generate detailed clinical reasoning
        clinical_reasoning = self._generate_clinical_reasoning(state)
        
        # Generate specific next steps
        next_steps = self._generate_next_steps(outcome, state)
        
        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(state)
        
        return {
            'outcome': outcome,
            'reasoning': state.reasoning,
            'clinical_reasoning': clinical_reasoning,
            'clinical_summary': clinical_summary,
            'next_steps': next_steps,
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
        """Extract variables needed for clinical assessment in hardcoded clinical workflow order"""
        clinical_variables = []
        
        # Hardcoded clinical workflow: diarrhea → fever/malaria → cough/pneumonia → malnutrition
        clinical_workflow_order = [
            # 1. Diarrhea Assessment
            'diarrhea', 'diarrhea_days', 'blood_in_stool',
            
            # 2. Fever/Malaria Assessment  
            'fever', 'temp', 'fever_days', 'malaria_area',
            
            # 3. Cough/Pneumonia Assessment
            'cough', 'cough_days', 'resp_rate', 'chest_indrawing',
            
            # 4. Malnutrition Assessment
            'muac_mm', 'swelling_both_feet', 'age_months'
        ]
        
        # Add variables from DMN that aren't already in danger signs
        if 'clinical_assessment' in self.decision_logic.decisions:
            decision = self.decision_logic.decisions['clinical_assessment']
            dmn_variables = []
            for input_col in decision.input_columns.values():
                if input_col.input_expression and input_col.input_expression not in self.danger_sign_variables:
                    dmn_variables.append(input_col.input_expression)
            
            # Merge with workflow order, prioritizing the clinical sequence
            for var in clinical_workflow_order:
                if var in dmn_variables and var not in clinical_variables:
                    clinical_variables.append(var)
            
            # Add any remaining DMN variables not in the workflow order
            for var in dmn_variables:
                if var not in clinical_variables:
                    clinical_variables.append(var)
        
        # Fallback to default clinical variables if DMN parsing fails
        if not clinical_variables:
            clinical_variables = clinical_workflow_order
        
        return clinical_variables
    
    def _get_next_clinical_variable(self, state: ConversationState) -> Optional[str]:
        """Intelligently select the next clinical variable based on current findings and clinical priority"""
        collected = state.collected_data
        
        # First, check if we already have enough data for a hospital referral
        if self._should_refer_to_hospital(collected):
            return None  # Stop asking questions, we have enough for hospital referral
        
        # Define clinical assessment groups with priority weights
        clinical_groups = {
            'danger_signs': {
                'vars': ['convulsion', 'unconscious', 'unable_to_drink', 'chest_indrawing', 'vomiting_everything'],
                'priority': 100,  # Highest priority
                'follow_up': []
            },
            'diarrhea': {
                'vars': ['diarrhea', 'diarrhea_days', 'blood_in_stool'],
                'priority': 80,
                'follow_up': ['dehydration_signs', 'muac_mm']  # If diarrhea present, check for dehydration
            },
            'fever_malaria': {
                'vars': ['fever', 'temp', 'fever_days', 'malaria_area'],
                'priority': 75,
                'follow_up': ['convulsion', 'unconscious']  # High fever can lead to convulsions
            },
            'respiratory': {
                'vars': ['cough', 'cough_days', 'resp_rate', 'chest_indrawing'],
                'priority': 70,
                'follow_up': ['unable_to_drink', 'fever']  # Respiratory distress can affect feeding
            },
            'malnutrition': {
                'vars': ['muac_mm', 'swelling_both_feet', 'age_months'],
                'priority': 60,
                'follow_up': ['diarrhea', 'fever']  # Malnutrition increases risk of infections
            }
        }
        
        # Find the highest priority group with missing variables
        for group_name, group_info in sorted(clinical_groups.items(), key=lambda x: x[1]['priority'], reverse=True):
            group_vars = group_info['vars']
            missing_vars = [v for v in group_vars if v not in collected]
            
            if missing_vars:
                # Check if we should ask follow-up questions first
                if group_info['follow_up']:
                    follow_up_missing = [v for v in group_info['follow_up'] if v not in collected]
                    if follow_up_missing:
                        return follow_up_missing[0]
                
                return missing_vars[0]
        
        # Ask any remaining clinical variables not in the main groups
        remaining_vars = [v for v in self.clinical_variables if v not in collected]
        if remaining_vars:
            return remaining_vars[0]
        
        return None  # All relevant clinical variables collected
    
    def _should_refer_to_hospital(self, collected_data: Dict[str, Any]) -> bool:
        """Check if we have enough data to recommend hospital referral immediately"""
        # Critical danger signs that require immediate hospital referral
        critical_signs = [
            'convulsion', 'unconscious', 'unable_to_drink', 'chest_indrawing', 
            'vomiting_everything', 'blood_in_stool'
        ]
        
        # Check if any critical signs are present
        for sign in critical_signs:
            if collected_data.get(sign) == True:
                return True
        
        # Check for severe dehydration (MUAC < 115mm)
        if collected_data.get('muac_mm', 0) > 0 and collected_data.get('muac_mm') < 115:
            return True
        
        # Check for severe respiratory distress (resp_rate > 50 for child)
        if collected_data.get('resp_rate', 0) > 50:
            return True
        
        # Check for high fever with other concerning symptoms
        if collected_data.get('temp', 0) >= 39.0 and collected_data.get('fever_days', 0) >= 3:
            return True
        
        return False
    
    def _should_refer_to_clinic(self, collected_data: Dict[str, Any]) -> bool:
        """Check if patient should be referred to clinic for further assessment"""
        # Moderate severity signs that require clinic assessment
        clinic_signs = [
            'fever', 'diarrhea', 'cough', 'swelling_both_feet'
        ]
        
        # Check for multiple moderate signs
        present_signs = sum(1 for sign in clinic_signs if collected_data.get(sign) == True)
        if present_signs >= 2:
            return True
        
        # Check for prolonged symptoms
        if collected_data.get('fever_days', 0) >= 3:
            return True
        if collected_data.get('diarrhea_days', 0) >= 3:
            return True
        if collected_data.get('cough_days', 0) >= 7:
            return True
        
        # Check for moderate dehydration (MUAC 115-125mm)
        if collected_data.get('muac_mm', 0) > 0 and 115 <= collected_data.get('muac_mm') <= 125:
            return True
        
        # Check for moderate respiratory distress (resp_rate 40-50)
        if collected_data.get('resp_rate', 0) > 0 and 40 <= collected_data.get('resp_rate') <= 50:
            return True
        
        return False
    
    def _is_safe_for_home_care(self, collected_data: Dict[str, Any]) -> bool:
        """Determine if patient is safe for home care after thorough assessment"""
        # Must have collected key clinical variables
        required_vars = ['temp', 'resp_rate', 'muac_mm', 'age_months']
        if not all(var in collected_data for var in required_vars):
            return False
        
        # Check for any concerning findings
        concerning_findings = [
            collected_data.get('fever') == True,
            collected_data.get('diarrhea') == True,
            collected_data.get('cough') == True,
            collected_data.get('temp', 0) >= 37.5,
            collected_data.get('resp_rate', 0) > 40,
            collected_data.get('muac_mm', 0) < 125,
            collected_data.get('swelling_both_feet') == True
        ]
        
        # If any concerning findings, not safe for home care
        if any(concerning_findings):
            return False
        
        return True
    
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
        """Evaluate clinical assessment and make final triage decision with thorough analysis"""
        collected = state.collected_data
        
        # First, double-check for any critical signs we might have missed
        if self._should_refer_to_hospital(collected):
            state.outcome = "Hospital"
            state.reasoning.append("Critical danger signs detected - immediate hospital referral required")
            state.flags['danger_sign'] = True
            return
        
        # Check for clinic referral criteria
        if self._should_refer_to_clinic(collected):
            state.outcome = "Clinic"
            state.reasoning.append("Clinical findings require clinic referral for further assessment")
            state.flags['clinic_referral'] = True
            return
        
        # Use DMN decision logic if available
        if 'clinical_assessment' in self.decision_logic.decisions:
            decision_table = self.decision_logic.decisions['clinical_assessment']
            matching_rules = self.dmn_parser.find_matching_rules(decision_table, collected)
            
            if matching_rules:
                rule = matching_rules[0]
                for output_id, value in rule.output_entries.items():
                    if value and value.strip():
                        triage_decision = value.strip().lower()
                        if triage_decision in ['hospital', 'clinic', 'home']:
                            state.outcome = triage_decision.title()
                            state.reasoning.append(f"DMN clinical assessment recommends: {state.outcome}")
                            
                            # Set appropriate flags
                            if triage_decision == 'clinic':
                                state.flags['clinic_referral'] = True
                            elif triage_decision == 'hospital':
                                state.flags['danger_sign'] = True
                            return
        
        # If we reach here, we need to be very thorough before recommending home care
        if self._is_safe_for_home_care(collected):
            state.outcome = "Home"
            state.reasoning.append("Thorough assessment completed - safe for home care with monitoring")
        else:
            # If we're not sure, err on the side of caution
            state.outcome = "Clinic"
            state.reasoning.append("Insufficient data for confident home care recommendation - clinic referral advised")
            state.flags['clinic_referral'] = True
    
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
    
    def _generate_clinical_reasoning(self, state: ConversationState) -> List[str]:
        """Generate detailed clinical reasoning based on collected data"""
        reasoning = []
        data = state.collected_data
        flags = state.flags
        
        # Check for danger signs
        danger_signs = []
        if data.get('convulsion'):
            danger_signs.append("convulsions present")
        if data.get('unconscious'):
            danger_signs.append("unconscious or unresponsive")
        if data.get('unable_to_drink') or data.get('cannot_drink_or_feed'):
            danger_signs.append("unable to drink or feed")
        if data.get('chest_indrawing'):
            danger_signs.append("severe chest indrawing")
        if data.get('vomiting_everything') or data.get('vomits_everything'):
            danger_signs.append("vomiting everything")
        
        if danger_signs:
            reasoning.append(f"⚠️ DANGER SIGNS IDENTIFIED: {', '.join(danger_signs)}")
            reasoning.append("These are serious warning signs requiring immediate medical attention")
        
        # Vital signs assessment
        if data.get('temp') and float(data['temp']) >= 38.5:
            reasoning.append(f"🌡️ High fever detected: {data['temp']}°C (normal <37.5°C)")
        if data.get('resp_rate'):
            resp_rate = float(data['resp_rate'])
            age_months = data.get('age_months', 12)
            age_months = float(age_months) if age_months else 12
            
            # WHO respiratory rate thresholds by age
            if age_months < 12 and resp_rate >= 50:
                reasoning.append(f"💨 Fast breathing: {resp_rate}/min (normal <50/min for age 2-11 months)")
            elif age_months >= 12 and resp_rate >= 40:
                reasoning.append(f"💨 Fast breathing: {resp_rate}/min (normal <40/min for age 12+ months)")
        
        # Malnutrition assessment
        if data.get('muac_mm'):
            muac = float(data['muac_mm'])
            if muac < 115:
                reasoning.append(f"📏 Severe acute malnutrition: MUAC {muac}mm (severe <115mm)")
            elif muac < 125:
                reasoning.append(f"📏 Moderate malnutrition: MUAC {muac}mm (moderate 115-124mm)")
        
        # Symptom duration assessment
        if data.get('diarrhea_days'):
            days = float(data['diarrhea_days'])
            if days >= 14:
                reasoning.append(f"💧 Persistent diarrhea: {days} days (concerning if ≥14 days)")
        
        if data.get('fever_days'):
            days = float(data['fever_days'])
            if days >= 7:
                reasoning.append(f"🔥 Prolonged fever: {days} days (concerning if ≥7 days)")
        
        if data.get('cough_days'):
            days = float(data['cough_days'])
            if days >= 14:
                reasoning.append(f"😷 Persistent cough: {days} days (concerning if ≥14 days)")
        
        # Blood in stool
        if data.get('blood_in_stool'):
            reasoning.append("🩸 Blood in stool indicates possible dysentery or serious intestinal condition")
        
        # Flag-based reasoning
        if flags.get('danger_sign'):
            reasoning.append("🚨 Critical danger signs require immediate hospital referral")
        elif flags.get('clinic_referral'):
            reasoning.append("🏪 Clinical assessment needed for proper evaluation and treatment")
        
        if not reasoning:
            reasoning.append("✅ No immediate danger signs or concerning symptoms identified")
            reasoning.append("📊 Vital signs and clinical assessment within normal ranges")
        
        return reasoning
    
    def _generate_clinical_summary(self, state: ConversationState) -> str:
        """Generate a concise clinical summary"""
        data = state.collected_data
        age_months = data.get('age_months', 'unknown')
        
        # Create age-appropriate summary
        if age_months != 'unknown':
            age_str = f"{age_months} months old"
        else:
            age_str = "child"
        
        # Identify primary concerns
        concerns = []
        if data.get('convulsion') or data.get('unconscious'):
            concerns.append("neurological emergency")
        if data.get('chest_indrawing') or (data.get('resp_rate') and float(data['resp_rate']) > 50):
            concerns.append("respiratory distress")
        if data.get('muac_mm') and float(data['muac_mm']) < 115:
            concerns.append("severe malnutrition")
        if data.get('blood_in_stool'):
            concerns.append("bloody diarrhea")
        if data.get('diarrhea_days') and float(data['diarrhea_days']) >= 14:
            concerns.append("persistent diarrhea")
        
        if concerns:
            return f"Clinical presentation: {age_str} presenting with {', '.join(concerns)}"
        else:
            return f"Clinical presentation: {age_str} with mild symptoms, no immediate danger signs"
    
    def _generate_next_steps(self, outcome: str, state: ConversationState) -> List[str]:
        """Generate specific next steps based on triage outcome"""
        next_steps = []
        data = state.collected_data
        
        if outcome == "Hospital":
            next_steps.extend([
                "🚨 URGENT: Transport to hospital immediately",
                "⏰ Do not delay - accompany the child to ensure immediate medical attention",
                "📋 Bring this assessment summary and any medications the child is taking",
                "🩺 Emergency priority: inform hospital staff of danger signs identified"
            ])
            
            # Specific emergency management
            if data.get('convulsion'):
                next_steps.append("⚡ If convulsions recur: protect from injury, do not restrain, clear airway")
            if data.get('unconscious'):
                next_steps.append("😴 Monitor breathing, maintain clear airway, recovery position if possible")
            if data.get('chest_indrawing'):
                next_steps.append("💨 Keep child upright, do not lay flat, monitor breathing continuously")
            if data.get('unable_to_drink'):
                next_steps.append("💧 Do not force feeding - hospital will manage fluid replacement")
            
        elif outcome == "Clinic":
            next_steps.extend([
                "🏪 Refer to nearest health facility within 24 hours",
                "📋 Bring this assessment summary for healthcare provider review",
                "⏰ Monitor symptoms closely - return immediately if condition worsens"
            ])
            
            # Specific clinic management
            if data.get('muac_mm') and float(data['muac_mm']) < 125:
                next_steps.append("📏 Nutritional assessment and therapeutic feeding program enrollment needed")
            if data.get('diarrhea_days') and float(data['diarrhea_days']) >= 7:
                next_steps.append("💧 Stool examination and targeted treatment required")
            if data.get('fever_days') and float(data['fever_days']) >= 3:
                next_steps.append("🔥 Malaria testing and fever investigation needed")
            
            # Warning signs to watch for
            next_steps.extend([
                "⚠️ Return immediately if: convulsions, unconsciousness, unable to drink, worsening breathing",
                "📞 Seek advice if symptoms persist or worsen after clinic visit"
            ])
            
        else:  # Home care
            next_steps.extend([
                "🏠 Home care appropriate with close monitoring",
                "👥 Educate caregiver on warning signs and supportive care",
                "📊 Follow-up assessment in 2-3 days or if symptoms change"
            ])
            
            # Specific home care instructions
            if data.get('diarrhea') or data.get('diarrhea_days'):
                next_steps.extend([
                    "💧 Continue breastfeeding and increase fluids (ORS if available)",
                    "🥄 Give zinc supplementation if available (10mg for <6 months, 20mg for 6+ months)"
                ])
            
            if data.get('fever'):
                next_steps.extend([
                    "🌡️ Paracetamol for fever relief (avoid aspirin in children)",
                    "🧊 Tepid sponging for high fever, dress lightly"
                ])
            
            if data.get('cough'):
                next_steps.append("😷 Honey for cough relief (only if >12 months old), increase fluids")
            
            # Universal home care advice
            next_steps.extend([
                "🍽️ Maintain nutrition - continue normal feeding plus extra fluids",
                "😴 Ensure adequate rest and comfort measures",
                "🧼 Good hygiene practices to prevent spread to others"
            ])
            
            # Warning signs for immediate return
            next_steps.extend([
                "🚨 DANGER SIGNS - seek immediate help if child develops:",
                "   • Convulsions or fits",
                "   • Becomes unconscious or very sleepy",
                "   • Cannot drink or breastfeed",
                "   • Severe difficulty breathing or chest pulling in",
                "   • Vomits everything",
                "📞 Contact health facility immediately if any danger signs appear"
            ])
        
        return next_steps
    
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