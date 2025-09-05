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
  "rule_id": "MODULE-XX", 
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
    "guideline_ref": "WHO-IMCI-2014-MODULE-XX",
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
                        {"role": "system", "content": self.system_prompt},
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
