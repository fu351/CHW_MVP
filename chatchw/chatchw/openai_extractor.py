"""
OpenAI-powered rule extraction from PDFs and text.
Uses OpenAI API to intelligently extract clinical rules and convert them to structured format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from openai import OpenAI

class OpenAIRuleExtractor:
    """Extract clinical rules using OpenAI API for intelligent analysis."""
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """Initialize with OpenAI API key and optional custom system prompt."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # System prompt for rule extraction (can be overridden)
        self.system_prompt = system_prompt or """You are a clinical rule extraction expert. Your job is to analyze WHO Community Health Worker (CHW) guidelines and extract structured clinical decision rules.

For each clinical condition or scenario you find, create a rule with this exact JSON structure:

{
  "rule_id": "MOD-XX", 
  "when": [
    {
      "obs": "temp|resp_rate|muac_mm",
      "op": "eq|lt|le|gt|ge", 
      "value": number
    }
    OR
    {
      "sym": "feels_very_hot|blood_in_stool|diarrhea_days|convulsion|edema_both_feet",
      "eq": true|false
    }
    OR
    {
      "any_of": [condition_list]
    }
    OR  
    {
      "all_of": [condition_list]
    }
  ],
  "then": {
    "propose_triage": "hospital|clinic|home",
    "set_flags": ["danger.sign", "fever.high", etc],
    "reasons": ["descriptive_reason"],
    "actions": [{"id": "action_name", "if_available": true|false}],
    "guideline_ref": "WHO-IMCI-2014-MOD-XX",
    "priority": number (1-100, higher = more urgent)
  }
}

IMPORTANT MAPPINGS:
- Temperature conditions → "obs": "temp", "op": "ge|gt|le|lt", "value": temp_in_celsius
- MUAC measurements → "obs": "muac_mm", "op": "lt|le", "value": measurement_in_mm  
- Respiratory rate → "obs": "resp_rate", "op": "gt|ge", "value": breaths_per_minute
- Symptoms like fever, convulsion, blood in stool → "sym": "symptom_name", "eq": true
- Absence of symptoms → "sym": "symptom_name", "eq": false
- Danger signs should have priority ≥ 90 and "set_flags": ["danger.sign"]
- Hospital referrals should have priority ≥ 80
- Clinic referrals should have priority 30-79
- Home care should have priority < 30

Extract ALL clinical decision rules you can find. Be comprehensive and thorough."""

        # Intelligent DMN generation system prompt
        self.dmn_generation_prompt = """You are an expert DMN (Decision Model and Notation) designer specializing in clinical decision support systems. Your task is to create a clean, logical, and non-overlapping DMN decision table from extracted clinical rules.

CRITICAL REQUIREMENTS:
1. **PRIORITY-BASED RULES**: Use hitPolicy="FIRST" - rules are evaluated in order, first match wins
2. **LOGICAL PRIORITIZATION**: Hospital > Clinic > Home care decisions (danger signs first!)
3. **COMPREHENSIVE COVERAGE**: Cover all clinical scenarios from the extracted rules
4. **CLEAN STRUCTURE**: Use only essential input variables, add final catch-all rule

INPUT: You will receive extracted clinical rules in JSON format.

OUTPUT: Generate a single DMN decision table with:
- **Input columns**: Only the most critical clinical variables
- **Output column**: Single "effect" column with format: "triage:HOSPITAL|CLINIC|HOME,flag:DANGER_SIGN|CLINIC_REFERRAL,reason:brief_reason"
- **Rules**: Non-overlapping, prioritized clinical decision rules

DMN STRUCTURE:
```xml
<dmn:definitions xmlns:dmn="https://www.omg.org/spec/DMN/20191111/MODEL/" id="Defs_1" name="ChatCHW Clinical Decision Support" namespace="chatchw">
  <dmn:inputData id="input_[variable]" name="[variable]">
    <dmn:variable name="[variable]" typeRef="string" />
  </dmn:inputData>
  
  <dmn:decision id="clinical_assessment" name="Clinical Assessment">
    <dmn:decisionTable id="decision_table_1" hitPolicy="FIRST">
      <dmn:input id="input_[var1]" label="[var1]">
        <dmn:inputExpression typeRef="string">
          <dmn:text>[var1]</dmn:text>
        </dmn:inputExpression>
      </dmn:input>
      <dmn:output id="output_effect" label="effect" typeRef="string" />
      
      <dmn:rule id="rule_1">
        <dmn:inputEntry id="input_entry_1">
          <dmn:text>="true"</dmn:text>
        </dmn:inputEntry>
        <dmn:outputEntry id="output_entry_1">
          <dmn:text>"triage:HOSPITAL,flag:DANGER_SIGN,reason:critical_danger_sign"</dmn:text>
        </dmn:outputEntry>
      </dmn:rule>
    </dmn:decisionTable>
  </dmn:decision>
</dmn:definitions>
```

PRIORITIZATION LOGIC:
1. **HOSPITAL**: Critical danger signs, severe dehydration, high fever, severe respiratory distress
2. **CLINIC**: Moderate symptoms, prolonged illness, multiple symptoms, moderate dehydration
3. **HOME**: Mild symptoms, no danger signs, adequate hydration, normal vital signs

VARIABLE SELECTION:
Focus on these key clinical variables:
- convulsion, unconscious, unable_to_drink, chest_indrawing, vomiting_everything (danger signs)
- blood_in_stool, fever, temp, diarrhea, cough, resp_rate (symptoms)
- muac_mm, age_months (measurements)
- fever_days, diarrhea_days, cough_days (duration)

Generate a clean, logical DMN that eliminates rule overlaps and provides clear clinical decision paths."""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def analyze_text_with_openai(self, text: str, module_name: str = "extracted") -> List[Dict[str, Any]]:
        """Use OpenAI to analyze text and extract clinical rules."""
        try:
            # Split text into chunks if it's too long
            max_chunk_size = 12000  # Leave room for system prompt
            text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            all_rules = []
            
            for i, chunk in enumerate(text_chunks):
                user_prompt = f"""Analyze this WHO CHW guideline text and extract ALL clinical decision rules. 
                
Module name: {module_name}
Text chunk {i+1}/{len(text_chunks)}:

{chunk}

Return ONLY a valid JSON array of rule objects. No other text or explanation."""

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Use the more cost-effective model
                    messages=[
                        {"role": "system", "content": self.extraction_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=4000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Clean up response - remove markdown formatting if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                try:
                    chunk_rules = json.loads(response_text)
                    if isinstance(chunk_rules, list):
                        # Add rule IDs if missing
                        for j, rule in enumerate(chunk_rules):
                            if "rule_id" not in rule or not rule["rule_id"]:
                                rule["rule_id"] = f"{module_name.upper()}-{len(all_rules) + j + 1:02d}"
                        all_rules.extend(chunk_rules)
                    elif isinstance(chunk_rules, dict):
                        # Single rule returned
                        if "rule_id" not in chunk_rules or not chunk_rules["rule_id"]:
                            chunk_rules["rule_id"] = f"{module_name.upper()}-{len(all_rules) + 1:02d}"
                        all_rules.append(chunk_rules)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON from chunk {i+1}: {e}")
                    print(f"Response was: {response_text[:200]}...")
                    continue
            
            return all_rules
            
        except Exception as e:
            raise RuntimeError(f"OpenAI analysis failed: {e}")

    def extract_rules_from_pdf(self, pdf_path: str, module_name: str = "extracted") -> List[Dict[str, Any]]:
        """Extract rules from PDF using OpenAI analysis."""
        print(f"🔍 Extracting text from PDF: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        
        print(f"📄 Extracted {len(text)} characters of text")
        print(f"🤖 Analyzing with OpenAI (module: {module_name})...")
        
        rules = self.analyze_text_with_openai(text, module_name)
        
        print(f"✅ Extracted {len(rules)} clinical rules")
        return rules

    def extract_rules_from_text(self, text: str, module_name: str = "text_rules") -> List[Dict[str, Any]]:
        """Extract rules from raw text using OpenAI analysis."""
        print(f"🤖 Analyzing text with OpenAI (module: {module_name})...")
        
        rules = self.analyze_text_with_openai(text, module_name)
        
        print(f"✅ Extracted {len(rules)} clinical rules")
        return rules

    def validate_and_clean_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean up extracted rules."""
        cleaned_rules = []
        
        for i, rule in enumerate(rules):
            try:
                # Ensure required fields exist
                if "rule_id" not in rule:
                    rule["rule_id"] = f"RULE-{i+1:02d}"
                
                if "when" not in rule:
                    rule["when"] = []
                
                # Fix when field - ensure it's always an array
                if isinstance(rule["when"], dict):
                    rule["when"] = [rule["when"]]
                elif not isinstance(rule["when"], list):
                    rule["when"] = []
                
                # Validate and fix conditions in when array
                fixed_conditions = []
                for condition in rule["when"]:
                    if isinstance(condition, dict):
                        fixed_condition = self._fix_condition(condition)
                        if fixed_condition:
                            fixed_conditions.append(fixed_condition)
                
                rule["when"] = fixed_conditions
                
                if "then" not in rule:
                    rule["then"] = {}
                
                # Ensure 'then' has required structure
                then_clause = rule["then"]
                if "priority" not in then_clause:
                    then_clause["priority"] = 50  # Default priority
                
                if "guideline_ref" not in then_clause:
                    then_clause["guideline_ref"] = f"WHO-IMCI-2014-{rule['rule_id']}"
                
                # Only add rules that have valid conditions
                if rule["when"]:
                    cleaned_rules.append(rule)
                
            except Exception as e:
                print(f"Warning: Skipping invalid rule {i}: {e}")
                continue
        
        return cleaned_rules

    def _fix_condition(self, condition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix and validate a single condition."""
        try:
            # Handle observation conditions
            if "obs" in condition and "op" in condition and "value" in condition:
                return condition
            
            # Handle symptom conditions
            if "sym" in condition:
                # Fix malformed symptom conditions
                if "ge" in condition or "gt" in condition or "lt" in condition or "le" in condition:
                    # This should be an observation, not a symptom
                    op = None
                    value = None
                    if "ge" in condition:
                        op, value = "ge", condition["ge"]
                    elif "gt" in condition:
                        op, value = "gt", condition["gt"]
                    elif "lt" in condition:
                        op, value = "lt", condition["lt"]
                    elif "le" in condition:
                        op, value = "le", condition["le"]
                    
                    # Convert symptom to observation
                    obs_name = condition["sym"]
                    if obs_name == "diarrhea_days":
                        obs_name = "diarrhea_duration_days"
                    
                    return {
                        "obs": obs_name,
                        "op": op,
                        "value": value
                    }
                
                # Ensure eq field exists for symptom conditions
                if "eq" not in condition:
                    condition["eq"] = True
                
                return condition
            
            # Handle complex conditions
            if "any_of" in condition or "all_of" in condition:
                return condition
            
            return None
            
        except Exception as e:
            print(f"Warning: Could not fix condition: {e}")
            return None

    def process_pdf_to_rules(self, pdf_path: str, module_name: str = "extracted") -> List[Dict[str, Any]]:
        """Complete pipeline: PDF → OpenAI analysis → validated rules."""
        rules = self.extract_rules_from_pdf(pdf_path, module_name)
        return self.validate_and_clean_rules(rules)

    def process_text_to_rules(self, text: str, module_name: str = "text_rules") -> List[Dict[str, Any]]:
        """Complete pipeline: Text → OpenAI analysis → validated rules."""
        rules = self.extract_rules_from_text(text, module_name)
        return self.validate_and_clean_rules(rules)

    def generate_intelligent_dmn(self, rules_json: List[Dict[str, Any]], module_name: str = "WHO_CHW") -> str:
        """Generate an intelligent, clean DMN from extracted rules using OpenAI."""
        print(f"🤖 Generating intelligent DMN from {len(rules_json)} rules...")
        
        # Reduce rules to prevent token overflow - use top priority rules only
        prioritized_rules = sorted(rules_json, key=lambda x: x.get('then', {}).get('priority', 0), reverse=True)
        selected_rules = prioritized_rules[:30]  # Use only top 30 high-priority rules
        
        # Prepare the user prompt with selected rules
        user_prompt = f"""EXTRACTED_CLINICAL_RULES (Top {len(selected_rules)} Priority Rules):
{json.dumps(selected_rules, indent=2)}

MODULE_NAME: {module_name}

Generate a clean, logical DMN decision table that:
1. Eliminates all rule overlaps (maintains UNIQUE hit policy)
2. Prioritizes clinical decisions (Hospital > Clinic > Home)
3. Uses only essential input variables
4. Covers all clinical scenarios from the rules
5. Provides clear, non-conflicting decision paths

Focus on creating a single, comprehensive decision table that can drive the chatbot's clinical assessment.

RULE ORDERING STRATEGY:
1. Start with the most critical danger signs (convulsion, unconscious, etc.)
2. Add other hospital referral criteria
3. Add clinic referral criteria  
4. End with a catch-all rule: all inputs "-" → "triage:home,reason:no_concerns_identified"

IMPORTANT: 
1. Ensure the XML is complete with proper closing tags
2. Properly escape XML characters: use &lt; for < and &gt; for > in dmn:text elements
3. Use FEEL expressions like "< 12" should be "&lt; 12" in XML

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.dmn_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=12000  # Increased to allow complete XML generation
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract DMN XML from response
            dmn_xml = self._extract_dmn_xml(response_text)
            
            if dmn_xml:
                print(f"✅ Generated intelligent DMN with clean decision logic")
                return dmn_xml
            else:
                raise RuntimeError("Failed to extract DMN XML from OpenAI response")
            
        except Exception as e:
            raise RuntimeError(f"OpenAI DMN generation failed: {e}")

    def _extract_dmn_xml(self, response_text: str) -> str:
        """Extract DMN XML from OpenAI response."""
        # Look for DMN XML in code blocks first
        xml_block_start = response_text.find("```xml")
        if xml_block_start != -1:
            # XML is in a code block
            dmn_start = response_text.find("<dmn:definitions", xml_block_start)
            if dmn_start != -1:
                dmn_end = response_text.find("</dmn:definitions>", dmn_start)
                if dmn_end != -1:
                    dmn_end += len("</dmn:definitions>")
                    return response_text[dmn_start:dmn_end].strip()
                else:
                    # Check if response was truncated
                    print(f"⚠️  DMN XML appears to be truncated. Response ends with: {response_text[-100:]}")
                    return None
        
        # Fallback: look for direct XML without code blocks
        dmn_start = response_text.find("<dmn:definitions")
        if dmn_start != -1:
            dmn_end = response_text.find("</dmn:definitions>", dmn_start)
            if dmn_end != -1:
                dmn_end += len("</dmn:definitions>")
                return response_text[dmn_start:dmn_end].strip()
            else:
                print(f"⚠️  DMN XML appears to be truncated. Response ends with: {response_text[-100:]}")
                return None
        
        print(f"⚠️  No DMN XML found in response. Response preview: {response_text[:500]}")
        return None

    def generate_related_bpmn_dmn(self, rules_json: List[Dict[str, Any]], module_name: str = "WHO_CHW") -> Dict[str, str]:
        """Generate both BPMN and DMN together so OpenAI can relate them intelligently."""
        print(f"🤖 Generating related BPMN+DMN from {len(rules_json)} rules...")
        
        # Enhanced system prompt for related generation
        related_generation_prompt = """You are an expert in clinical decision support systems. Generate BOTH a BPMN workflow and DMN decision table that work together seamlessly for a medical chatbot.

CRITICAL REQUIREMENTS:
1. **BPMN-DMN INTEGRATION**: The BPMN must reference DMN decisions and the DMN must support the BPMN flow
2. **INTELLIGENT CHAT FLOW**: Design a logical conversation flow that follows clinical priorities
3. **NO RULE OVERLAPS**: DMN must maintain hitPolicy="UNIQUE"
4. **CLINICAL PRIORITIZATION**: Hospital > Clinic > Home care decisions

BPMN REQUIREMENTS:
- Start with danger signs assessment
- Flow through clinical assessment modules (diarrhea → fever → respiratory → malnutrition)
- Use BusinessRuleTask elements that reference DMN decisions
- Include proper gateways with conditions based on DMN outputs
- End with three outcomes: Hospital, Clinic, Home

DMN REQUIREMENTS:
- Single decision table with hitPolicy="UNIQUE"
- Input variables that match BPMN flow needs
- Output format: "triage:HOSPITAL|CLINIC|HOME,flag:DANGER_SIGN|CLINIC_REFERRAL,reason:brief_reason"
- Prioritize critical conditions (hospital) over moderate (clinic) over mild (home)

INTEGRATION POINTS:
- BPMN BusinessRuleTask should reference DMN decision IDs
- DMN input variables should match BPMN data collection points
- BPMN gateway conditions should use DMN output flags

Generate both artifacts that work together as a cohesive clinical decision support system."""

        user_prompt = f"""EXTRACTED_CLINICAL_RULES:
{json.dumps(rules_json, indent=2)[:8000]}  # Truncate for token limits

MODULE_NAME: {module_name}

Generate a complete BPMN workflow and DMN decision table that work together for intelligent medical triage. The BPMN should define the conversation flow, and the DMN should provide the clinical decision logic.

Focus on creating a seamless integration where:
1. BPMN guides the conversation flow
2. DMN makes clinical decisions at key points
3. The flow is logical and follows clinical priorities
4. No rule overlaps exist in the DMN"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": related_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=8000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract both BPMN and DMN XML
            bpmn_xml = self._extract_bpmn_xml(response_text)
            dmn_xml = self._extract_dmn_xml(response_text)
            
            if bpmn_xml and dmn_xml:
                print(f"✅ Generated related BPMN+DMN with intelligent integration")
                return {
                    "bpmn_xml": bpmn_xml,
                    "dmn_xml": dmn_xml,
                    "raw_response": response_text
                }
            else:
                raise RuntimeError("Failed to extract both BPMN and DMN XML from OpenAI response")
            
        except Exception as e:
            raise RuntimeError(f"OpenAI related BPMN+DMN generation failed: {e}")

    def generate_sequential_bpmn_dmn(self, rules_json: List[Dict[str, Any]], module_name: str = "WHO_CHW") -> Dict[str, str]:
        """Generate BPMN first, then DMN with BPMN context for intelligent flow logic."""
        print(f"🤖 Generating sequential BPMN→DMN from {len(rules_json)} rules...")
        
        # Step 1: Generate BPMN first
        bpmn_prompt = """You are a BPMN workflow designer for medical chatbots. Generate a BPMN 2.0 process that defines the conversation flow for clinical assessment.

REQUIREMENTS:
- Start with danger signs assessment
- Flow through clinical modules: diarrhea → fever/malaria → respiratory → malnutrition
- Use UserTask for data collection, BusinessRuleTask for decisions
- Include proper gateways and sequence flows
- End with three outcomes: Hospital, Clinic, Home
- Use proper BPMN 2.0 syntax with namespaces

Generate a clean BPMN that defines the conversation structure."""

        try:
            # Generate BPMN
            bpmn_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": bpmn_prompt},
                    {"role": "user", "content": f"Generate BPMN for clinical assessment using these rules:\n{json.dumps(rules_json[:20], indent=2)}"}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            bpmn_xml = self._extract_bpmn_xml(bpmn_response.choices[0].message.content.strip())
            
            if not bpmn_xml:
                raise RuntimeError("Failed to generate BPMN")
            
            print("✅ Generated BPMN workflow")
            
            # Step 2: Generate DMN with BPMN context
            dmn_with_context_prompt = f"""You are a DMN designer. Generate a decision table that works with the provided BPMN workflow.

BPMN CONTEXT:
{bpmn_xml[:2000]}  # Truncate for context

CLINICAL RULES:
{json.dumps(rules_json, indent=2)[:4000]}  # Truncate for token limits

REQUIREMENTS:
1. **BPMN INTEGRATION**: DMN must support the BPMN workflow structure
2. **NO OVERLAPS**: Maintain hitPolicy="UNIQUE" - no conflicting rules
3. **CLINICAL PRIORITY**: Hospital > Clinic > Home decisions
4. **INPUT VARIABLES**: Match the data collection points in the BPMN
5. **OUTPUT FORMAT**: "triage:HOSPITAL|CLINIC|HOME,flag:DANGER_SIGN|CLINIC_REFERRAL,reason:brief_reason"

Generate a DMN decision table that seamlessly integrates with the BPMN workflow."""

            dmn_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": dmn_with_context_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            dmn_xml = self._extract_dmn_xml(dmn_response.choices[0].message.content.strip())
            
            if not dmn_xml:
                raise RuntimeError("Failed to generate DMN")
            
            print("✅ Generated DMN with BPMN context")
            
            return {
                "bpmn_xml": bpmn_xml,
                "dmn_xml": dmn_xml,
                "generation_method": "sequential"
            }
            
        except Exception as e:
            raise RuntimeError(f"OpenAI sequential BPMN→DMN generation failed: {e}")

    def _extract_bpmn_xml(self, response_text: str) -> str:
        """Extract BPMN XML from OpenAI response."""
        # Look for BPMN XML in the response
        bpmn_start = response_text.find("```xml")
        if bpmn_start == -1:
            bpmn_start = response_text.find("<bpmn:definitions")
        
        if bpmn_start != -1:
            if bpmn_start != response_text.find("```xml"):
                # Direct XML without code block
                bpmn_end = response_text.find("</bpmn:definitions>", bpmn_start)
                if bpmn_end != -1:
                    bpmn_end += len("</bpmn:definitions>")
                    return response_text[bpmn_start:bpmn_end].strip()
            else:
                # XML in code block
                bpmn_start = response_text.find("<bpmn:definitions", bpmn_start)
                if bpmn_start != -1:
                    bpmn_end = response_text.find("</bpmn:definitions>", bpmn_start)
                    if bpmn_end != -1:
                        bpmn_end = response_text.find("\n```", bpmn_end)
                        if bpmn_end == -1:
                            bpmn_end = response_text.find("</bpmn:definitions>", bpmn_start) + len("</bpmn:definitions>")
                        return response_text[bpmn_start:bpmn_end].strip()
        
        return None

