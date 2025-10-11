"""
Text-to-flowchart processor.
Handles direct text input and converts it to clinical workflows.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .openai_extractor import OpenAIRuleExtractor
from .flowchart_generator import FlowchartGenerator
from .bpmn_builder import build_bpmn
from .dmn_builder import generate_dmn


class TextToFlowchartProcessor:
    """Process text directly into clinical flowcharts and workflow artifacts."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize with optional OpenAI API key for enhanced rule extraction."""
        self.openai_extractor = None
        if openai_api_key:
            try:
                self.openai_extractor = OpenAIRuleExtractor(openai_api_key)
            except Exception as e:
                print(f"Warning: OpenAI extractor not available: {e}")
        
        self.flowchart_generator = FlowchartGenerator()

    def process_text_to_rules(self, text: str, module_name: str = "text_input", use_openai: bool = True) -> List[Dict[str, Any]]:
        """Extract clinical rules from text input."""
        
        if use_openai and self.openai_extractor:
            print("🤖 Using OpenAI for intelligent rule extraction...")
            return self.openai_extractor.process_text_to_rules(text, module_name)
        else:
            print("📝 Using basic pattern-based rule extraction...")
            return self._basic_text_extraction(text, module_name)

    def _basic_text_extraction(self, text: str, module_name: str) -> List[Dict[str, Any]]:
        """Basic rule extraction without OpenAI (fallback method)."""
        import re
        
        rules = []
        rule_counter = 1
        
        # Simple patterns for common clinical conditions
        patterns = {
            r"temperature.*?≥\s*(\d+\.?\d*)|temp.*?≥\s*(\d+\.?\d*)|fever.*?≥\s*(\d+\.?\d*)": {
                "condition": {"obs": "temp", "op": "ge", "value": "TEMP"},
                "action": {"propose_triage": "clinic", "reasons": ["High fever"], "priority": 70}
            },
            r"convulsion|seizure": {
                "condition": {"sym": "convulsion", "eq": True},
                "action": {"propose_triage": "hospital", "set_flags": ["danger.sign"], "reasons": ["Convulsion present"], "priority": 100}
            },
            r"blood.*?stool|bloody.*?stool": {
                "condition": {"sym": "blood_in_stool", "eq": True},
                "action": {"propose_triage": "clinic", "reasons": ["Blood in stool"], "priority": 60}
            },
            r"MUAC.*?<\s*(\d+)|muac.*?<\s*(\d+)": {
                "condition": {"obs": "muac_mm", "op": "lt", "value": "MUAC"},
                "action": {"propose_triage": "clinic", "reasons": ["Malnutrition risk"], "priority": 80}
            },
            r"edema.*?feet|swelling.*?feet": {
                "condition": {"sym": "edema_both_feet", "eq": True},
                "action": {"propose_triage": "hospital", "set_flags": ["danger.sign"], "reasons": ["Severe malnutrition"], "priority": 95}
            }
        }
        
        for pattern, rule_template in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract numeric values if present
                condition = rule_template["condition"].copy()
                if "TEMP" in str(condition.get("value")):
                    temp_value = None
                    for group in match.groups():
                        if group:
                            temp_value = float(group)
                            break
                    if temp_value:
                        condition["value"] = temp_value
                    else:
                        condition["value"] = 38.5  # Default fever threshold
                
                elif "MUAC" in str(condition.get("value")):
                    muac_value = None
                    for group in match.groups():
                        if group:
                            muac_value = int(group)
                            break
                    if muac_value:
                        condition["value"] = muac_value
                    else:
                        condition["value"] = 115  # Default MUAC threshold
                
                rule = {
                    "rule_id": f"{module_name.upper()}-{rule_counter:02d}",
                    "when": [condition],
                    "then": {
                        **rule_template["action"],
                        "guideline_ref": f"TEXT-{module_name.upper()}-{rule_counter:02d}"
                    }
                }
                
                rules.append(rule)
                rule_counter += 1
        
        return rules

    def create_comprehensive_workflow(self, 
                                    text: str, 
                                    module_name: str = "text_workflow",
                                    output_dir: str = "text_workflow_output",
                                    use_openai: bool = True) -> Dict[str, str]:
        """Create complete workflow from text input."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create organized subdirectories
        (output_path / "01_extracted_data").mkdir(exist_ok=True)
        (output_path / "02_process_models").mkdir(exist_ok=True) 
        (output_path / "03_readable_formats").mkdir(exist_ok=True)
        (output_path / "04_clinical_flowcharts").mkdir(exist_ok=True)
        (output_path / "05_technical_diagrams").mkdir(exist_ok=True)
        
        generated_files = {}
        
        try:
            # Step 1: Extract rules from text
            print(f"🔍 Processing text input (length: {len(text)} chars)...")
            rules = self.process_text_to_rules(text, module_name, use_openai)
            
            # Save extracted rules
            rules_file = output_path / "01_extracted_data" / "extracted_rules_from_text.json"
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=2)
            generated_files["rules"] = str(rules_file)
            print(f"✅ Extracted {len(rules)} rules → {rules_file}")
            
            if not rules:
                print("⚠️  No rules extracted. Creating placeholder workflow...")
                return generated_files
            
            rulepacks = {module_name: rules}
            
            # Step 2: Generate BPMN
            print("🔧 Generating BPMN...")
            bpmn_xml = build_bpmn(rulepacks)
            bpmn_file = output_path / "02_process_models" / "text_workflow_process.bpmn"
            bpmn_file.write_text(bpmn_xml, encoding='utf-8')
            generated_files["bpmn"] = str(bpmn_file)
            print(f"✅ Generated BPMN → {bpmn_file}")
            
            # Step 3: Generate DMN
            print("🔧 Generating DMN...")
            dmn_xml = generate_dmn(rulepacks)
            dmn_file = output_path / "02_process_models" / "text_decision_logic.dmn"
            dmn_file.write_text(dmn_xml, encoding='utf-8')
            generated_files["dmn"] = str(dmn_file)
            print(f"✅ Generated DMN → {dmn_file}")
            
            # Step 4: Convert to readable JSON formats
            try:
                bpmn_json = output_path / "03_readable_formats" / "workflow_process_readable.json"
                self.flowchart_generator.save_json_format(str(bpmn_file), str(bpmn_json), "bpmn")
                generated_files["bpmn_json"] = str(bpmn_json)
                print(f"✅ Converted BPMN to JSON → {bpmn_json}")
            except Exception as e:
                print(f"⚠️  BPMN JSON conversion failed: {e}")
            
            try:
                dmn_json = output_path / "03_readable_formats" / "decision_logic_readable.json"
                self.flowchart_generator.save_json_format(str(dmn_file), str(dmn_json), "dmn")
                generated_files["dmn_json"] = str(dmn_json)
                print(f"✅ Converted DMN to JSON → {dmn_json}")
            except Exception as e:
                print(f"⚠️  DMN JSON conversion failed: {e}")
            
            # Step 5: Generate clinical flowchart (main user-facing workflow)
            try:
                clinical_flowchart = output_path / "04_clinical_flowcharts" / "clinical_workflow_guide.png"
                self.flowchart_generator.generate_clinical_flowchart(rules, f"Text-Based {module_name}", str(clinical_flowchart))
                generated_files["clinical_flowchart"] = str(clinical_flowchart)
                print(f"✅ Generated clinical workflow guide → {clinical_flowchart}")
            except Exception as e:
                print(f"⚠️  Clinical flowchart generation failed: {e}")
            
            # Step 6: Generate technical flowcharts 
            try:
                bpmn_flowchart = output_path / "05_technical_diagrams" / "technical_bpmn_flowchart.png"
                self.flowchart_generator.generate_bpmn_flowchart(str(bpmn_file), str(bpmn_flowchart))
                generated_files["bpmn_flowchart"] = str(bpmn_flowchart)
                print(f"✅ Generated technical BPMN flowchart → {bpmn_flowchart}")
            except Exception as e:
                print(f"⚠️  Technical BPMN flowchart failed: {e}")
            
            try:
                dmn_flowchart = output_path / "05_technical_diagrams" / "technical_dmn_flowchart.png"
                self.flowchart_generator.generate_dmn_flowchart(str(dmn_file), str(dmn_flowchart))
                generated_files["dmn_flowchart"] = str(dmn_flowchart)
                print(f"✅ Generated technical DMN flowchart → {dmn_flowchart}")
            except Exception as e:
                print(f"⚠️  Technical DMN flowchart failed: {e}")
            
            print(f"\n🎉 Text-to-workflow processing complete!")
            print(f"📁 Generated {len(generated_files)} files in: {output_path}")
            
            return generated_files
            
        except Exception as e:
            print(f"❌ Text workflow processing failed: {e}")
            raise
