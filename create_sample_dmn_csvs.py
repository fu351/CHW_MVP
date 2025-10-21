#!/usr/bin/env python3
"""
Create sample DMN CSV files for testing the processor.
"""

import csv
import os
from pathlib import Path

def create_sample_csvs():
    """Create sample DMN module CSV files."""
    
    # Create the output directory
    output_dir = Path("sample_dmn_csvs")
    output_dir.mkdir(exist_ok=True)
    
    # Sample data for each module
    modules = {
        "dmn_module_a.csv": [
            {"rule_id": "A001", "condition": "age >= 65", "action": "refer_to_geriatric", "priority": 100},
            {"rule_id": "A002", "condition": "temperature > 38.5", "action": "check_fever", "priority": 50},
            {"rule_id": "A003", "condition": "blood_pressure > 140/90", "action": "monitor_bp", "priority": 75}
        ],
        "dmn_module_b.csv": [
            {"rule_id": "B001", "condition": "chest_pain = true", "action": "emergency_cardiac", "priority": 100},
            {"rule_id": "B002", "condition": "shortness_breath = true", "action": "respiratory_assessment", "priority": 80},
            {"rule_id": "B003", "condition": "dizziness = true", "action": "neurological_check", "priority": 60}
        ],
        "dmn_module_c.csv": [
            {"rule_id": "C001", "condition": "diabetes = true", "action": "glucose_monitoring", "priority": 70},
            {"rule_id": "C002", "condition": "hypertension = true", "action": "bp_management", "priority": 65},
            {"rule_id": "C003", "condition": "medication_allergy = true", "action": "allergy_alert", "priority": 90}
        ],
        "dmn_module_d.csv": [
            {"rule_id": "D001", "condition": "pregnancy = true", "action": "obstetric_care", "priority": 85},
            {"rule_id": "D002", "condition": "pediatric_age < 12", "action": "pediatric_protocol", "priority": 80},
            {"rule_id": "D003", "condition": "mental_health_crisis = true", "action": "psychiatric_referral", "priority": 95}
        ],
        "dmn_module_e.csv": [
            {"rule_id": "E001", "condition": "infection_signs = true", "action": "infection_control", "priority": 75},
            {"rule_id": "E002", "condition": "wound_care_needed = true", "action": "wound_management", "priority": 60},
            {"rule_id": "E003", "condition": "medication_review = true", "action": "pharmacy_consult", "priority": 55}
        ]
    }
    
    # Create each CSV file
    for filename, data in modules.items():
        filepath = output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        print(f"Created: {filepath}")
    
    # Create the XML configuration file
    config_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<instances>
    <instance id="dmn_module_a" src="jr://file-csv/dmn_module_a.csv"/>
    <instance id="dmn_module_b" src="jr://file-csv/dmn_module_b.csv"/>
    <instance id="dmn_module_c" src="jr://file-csv/dmn_module_c.csv"/>
    <instance id="dmn_module_d" src="jr://file-csv/dmn_module_d.csv"/>
    <instance id="dmn_module_e" src="jr://file-csv/dmn_module_e.csv"/>
    <instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
</instances>'''
    
    config_path = output_dir / "dmn_config.xml"
    with open(config_path, 'w') as f:
        f.write(config_xml)
    print(f"Created: {config_path}")
    
    print(f"\nSample CSV files created in: {output_dir}")
    print("You can now test the processor with:")
    print(f"python process_dmn_csv.py --base-dir {output_dir} --config {config_path}")

if __name__ == "__main__":
    create_sample_csvs()