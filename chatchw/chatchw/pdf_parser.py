from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class CHWRuleExtractor:
    """Extracts clinical rules from WHO CHW PDF documents."""
    
    def __init__(self):
        # Patterns for identifying clinical conditions and actions
        self.condition_patterns = [
            r"(?i)if\s+(.+?)\s+(?:then|:)",
            r"(?i)when\s+(.+?)\s+(?:then|:)",
            r"(?i)(.+?)\s+→\s+(.+)",
            r"(?i)(.+?)\s+->\s+(.+)",
            r"(?i)temperature\s+(?:>=|≥|>)\s*(\d+\.?\d*)",
            r"(?i)fever\s+(?:>=|≥|>)\s*(\d+\.?\d*)",
            r"(?i)MUAC\s+(?:<=|≤|<)\s*(\d+\.?\d*)",
            r"(?i)convulsion|seizure",
            r"(?i)blood\s+in\s+stool",
            r"(?i)diarrhea\s+(?:for\s+)?(\d+)\s+days?",
            r"(?i)edema\s+(?:in\s+)?both\s+feet",
        ]
        
        self.action_patterns = [
            r"(?i)refer\s+(?:to\s+)?hospital",
            r"(?i)refer\s+(?:to\s+)?clinic", 
            r"(?i)treat\s+at\s+home",
            r"(?i)give\s+(.+)",
            r"(?i)administer\s+(.+)",
            r"(?i)urgent\s+referral",
        ]
        
        self.danger_signs = [
            "convulsion", "seizure", "unable to drink", "vomiting everything",
            "lethargy", "unconscious", "severe dehydration"
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        if PdfReader is None:
            raise ImportError("pypdf not available. Install with: pip install pypdf")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Failed to read PDF {pdf_path}: {e}")

    def parse_temperature_condition(self, text: str) -> Optional[Dict]:
        """Parse temperature-based conditions."""
        temp_match = re.search(r"(?i)temperature\s+(?:>=|≥|>)\s*(\d+\.?\d*)", text)
        if temp_match:
            value = float(temp_match.group(1))
            return {
                "obs": "temp",
                "op": "ge",
                "value": value
            }
        return None

    def parse_muac_condition(self, text: str) -> Optional[Dict]:
        """Parse MUAC-based conditions."""
        muac_match = re.search(r"(?i)MUAC\s+(?:<=|≤|<)\s*(\d+\.?\d*)", text)
        if muac_match:
            value = float(muac_match.group(1))
            return {
                "obs": "muac_mm", 
                "op": "lt",
                "value": value
            }
        return None

    def parse_symptom_condition(self, text: str) -> Optional[Dict]:
        """Parse symptom-based conditions."""
        if re.search(r"(?i)convulsion|seizure", text):
            return {"sym": "convulsion", "eq": True}
        elif re.search(r"(?i)blood\s+in\s+stool", text):
            return {"sym": "blood_in_stool", "eq": True}
        elif re.search(r"(?i)very\s+hot|feels\s+hot", text):
            return {"sym": "feels_very_hot", "eq": True}
        elif re.search(r"(?i)edema\s+(?:in\s+)?both\s+feet", text):
            return {"sym": "edema_both_feet", "eq": True}
        
        diarrhea_match = re.search(r"(?i)diarrhea\s+(?:for\s+)?(\d+)\s+days?", text)
        if diarrhea_match:
            days = int(diarrhea_match.group(1))
            return {"sym": "diarrhea_days", "eq": days}
        
        return None

    def parse_action(self, text: str) -> Tuple[Optional[str], List[str], List[str]]:
        """Parse actions and determine triage level."""
        triage = None
        reasons = []
        flags = []
        
        text_lower = text.lower()
        
        # Check for danger signs
        for danger in self.danger_signs:
            if danger in text_lower:
                flags.append("danger.sign")
                triage = "hospital"
                reasons.append(f"{danger.replace(' ', '.')}.danger.sign")
                break
        
        # Check for referral patterns
        if re.search(r"(?i)refer\s+(?:to\s+)?hospital", text):
            triage = "hospital"
        elif re.search(r"(?i)refer\s+(?:to\s+)?clinic", text):
            triage = "clinic"
        elif re.search(r"(?i)treat\s+at\s+home", text):
            triage = "home"
        
        # Extract medication/treatment
        med_match = re.search(r"(?i)give\s+(.+?)(?:\.|,|$)", text)
        if med_match:
            med = med_match.group(1).strip()
            reasons.append(f"treatment.{med.replace(' ', '.')}")
            
        return triage, reasons, flags

    def extract_rules_from_text(self, text: str, module_name: str = "extracted") -> List[Dict]:
        """Extract rules from PDF text content."""
        rules = []
        
        # Split text into sections/paragraphs
        sections = re.split(r'\n\s*\n', text)
        
        rule_id_counter = 1
        
        for section in sections:
            section = section.strip()
            if len(section) < 50:  # Skip very short sections
                continue
                
            # Look for if-then patterns
            if_then_match = re.search(r"(?i)if\s+(.+?)\s+(?:then|:)\s*(.+)", section, re.DOTALL)
            if not if_then_match:
                if_then_match = re.search(r"(?i)(.+?)\s+→\s+(.+)", section)
            if not if_then_match:
                if_then_match = re.search(r"(?i)(.+?)\s+->\s+(.+)", section)
                
            if if_then_match:
                condition_text = if_then_match.group(1).strip()
                action_text = if_then_match.group(2).strip()
                
                # Parse condition
                conditions = []
                
                temp_cond = self.parse_temperature_condition(condition_text)
                if temp_cond:
                    conditions.append(temp_cond)
                    
                muac_cond = self.parse_muac_condition(condition_text)
                if muac_cond:
                    conditions.append(muac_cond)
                    
                sym_cond = self.parse_symptom_condition(condition_text)
                if sym_cond:
                    conditions.append(sym_cond)
                
                if conditions:
                    # Parse action
                    triage, reasons, flags = self.parse_action(action_text)
                    
                    # Determine priority based on flags
                    priority = 100 if "danger.sign" in flags else 10
                    
                    rule = {
                        "rule_id": f"{module_name.upper()}-{rule_id_counter:02d}",
                        "when": conditions,
                        "then": {
                            "guideline_ref": f"WHO-IMCI-2014-{module_name.upper()}-{rule_id_counter:02d}",
                            "priority": priority
                        }
                    }
                    
                    if flags:
                        rule["then"]["set_flags"] = flags
                    if triage:
                        rule["then"]["propose_triage"] = triage
                    if reasons:
                        rule["then"]["reasons"] = reasons
                        
                    rules.append(rule)
                    rule_id_counter += 1
        
        return rules

    def process_pdf_to_rules(self, pdf_path: str, module_name: str = "extracted") -> List[Dict]:
        """Main method to process a WHO CHW PDF and extract rules."""
        text = self.extract_text_from_pdf(pdf_path)
        rules = self.extract_rules_from_text(text, module_name)
        return rules
