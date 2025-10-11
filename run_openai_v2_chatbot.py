#!/usr/bin/env python3
"""Interactive ChatCHW with OpenAI v2 (212 comprehensive rules)."""

from chatchw.chatchw.chatbot_engine import ChatbotEngine

def main():
    print("🤖 ChatCHW v2 - OpenAI Enhanced (212 Rules)")
    print("=" * 60)
    
    # Load OpenAI v2 artifacts
    bpmn_path = "openai_output_v2/02_process_models/openai_workflow_process.bpmn"
    dmn_path = "openai_output_v2/02_process_models/openai_decision_logic.dmn"
    
    try:
        print(f"📋 Loading BPMN: {bpmn_path}")
        print(f"🧠 Loading DMN: {dmn_path}")
        
        engine = ChatbotEngine(bpmn_path, dmn_path)
        
        print(f"✅ OpenAI v2 engine loaded!")
        print(f"📱 Questions: {len(engine.questions)}")
        print(f"🧠 Decision tables: {len(engine.decision_logic.decisions)}")
        print(f"📊 Variables: {len(engine.decision_logic.variable_types)}")
        
        print(f"\n🏥 Starting WHO-Compliant Clinical Assessment...")
        print("=" * 50)
        print("(Type 'exit' or 'quit' to stop)")
        
        state = engine.start_conversation()
        
        # Interactive loop
        question_count = 0
        max_questions = 20
        
        while not engine.is_conversation_complete(state) and question_count < max_questions:
            question = engine.get_next_question(state)
            
            if not question:
                break
                
            print(f"\n❓ Question {question_count + 1}: {question.text}")
            print(f"   Variable: {question.variable} (Type: {question.question_type})")
            
            # Get input
            if question.question_type == 'boolean':
                while True:
                    answer = input("   Answer (yes/no): ").lower().strip()
                    if answer in ['yes', 'y', 'true', '1']:
                        value = True
                        break
                    elif answer in ['no', 'n', 'false', '0']:
                        value = False
                        break
                    elif answer in ['exit', 'quit']:
                        print("👋 Assessment stopped.")
                        return
                    else:
                        print("   Please answer yes/no")
                        
            elif question.question_type == 'numeric':
                while True:
                    try:
                        answer = input("   Answer (number): ").strip()
                        if answer.lower() in ['exit', 'quit']:
                            print("👋 Assessment stopped.")
                            return
                        value = float(answer)
                        break
                    except ValueError:
                        print("   Please enter a valid number")
                        
            else:
                answer = input("   Answer: ").strip()
                if answer.lower() in ['exit', 'quit']:
                    print("👋 Assessment stopped.")
                    return
                value = answer
            
            # Process answer
            state, valid = engine.process_answer(state, question.variable, value)
            
            if valid:
                print(f"   ✅ Recorded: {question.variable} = {value}")
            else:
                print(f"   ❌ Invalid input")
                continue
            
            question_count += 1
            
            # Emergency check
            if question.variable == 'convulsion' and value == True:
                print(f"\n🚨 DANGER SIGN DETECTED!")
                break
        
        # Final recommendation
        print(f"\n🏥 WHO-Compliant Assessment Complete")
        print("=" * 50)
        
        recommendation = engine.get_final_recommendation(state)
        
        outcome_emoji = {"Hospital": "🏥", "Clinic": "🏪", "Home": "🏠"}
        emoji = outcome_emoji.get(recommendation['outcome'], "📍")
        print(f"{emoji} TRIAGE DECISION: {recommendation['outcome'] or 'Continue Assessment'}")
        
        if recommendation['flags']:
            print(f"🚩 Clinical Flags: {', '.join(recommendation['flags'].keys())}")
        
        if recommendation['reasoning']:
            print(f"📋 Clinical Reasoning:")
            for reason in recommendation['reasoning'][:5]:
                print(f"   • {reason}")
        
        print(f"\n📊 Assessment Data:")
        for var, value in state.collected_data.items():
            print(f"   {var}: {value}")
        
        print(f"\n🎉 OpenAI v2 assessment complete!")
        print(f"Based on 212 comprehensive WHO-extracted rules")
        
    except FileNotFoundError:
        print(f"❌ Error: OpenAI v2 artifacts not found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
