#!/usr/bin/env python3
"""Utility to auto-generate questions from DMN variables."""

from chatchw.chatbot_engine import Question
from chatchw.dmn_parser import DMNParser

def generate_questions_from_dmn(dmn_file_path):
    """Auto-generate Question objects from DMN input variables."""
    parser = DMNParser()
    logic = parser.parse_dmn_file(dmn_file_path)
    
    questions = {}
    
    for decision in logic.decisions.values():
        for input_col in decision.input_columns.values():
            var_name = input_col.input_expression
            var_type = logic.variable_types.get(var_name, 'string')
            
            if var_name not in questions:
                # Generate human-readable question text
                question_text = generate_question_text(var_name, var_type)
                question_type = map_dmn_type_to_question_type(var_type)
                
                questions[var_name] = Question(var_name, question_text, question_type)
    
    return questions

def generate_question_text(var_name, var_type):
    """Generate human-readable question text from variable name."""
    # Convert snake_case to human readable
    words = var_name.replace('_', ' ').title()
    
    if var_type == 'boolean':
        if 'has' in var_name.lower() or 'is' in var_name.lower():
            return f"{words}?"
        else:
            return f"Does the patient have {words.lower()}?"
    elif var_type == 'number':
        if 'temp' in var_name.lower():
            return f"What is the patient's {words.lower()} (°C)?"
        elif 'days' in var_name.lower():
            return f"How many {words.lower()}?"
        elif 'mm' in var_name.lower():
            return f"What is the {words.lower()} measurement?"
        else:
            return f"What is the {words.lower()} value?"
    else:
        return f"What is the {words.lower()}?"

def map_dmn_type_to_question_type(dmn_type):
    """Map DMN variable type to Question type."""
    if dmn_type == 'boolean':
        return 'boolean'
    elif dmn_type in ['number', 'numeric']:
        return 'numeric'
    else:
        return 'string'

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        dmn_path = sys.argv[1]
        questions = generate_questions_from_dmn(dmn_path)
        
        print(f"Generated {len(questions)} questions:")
        for var, q in questions.items():
            print(f"  {var}: {q.text} ({q.question_type})")
