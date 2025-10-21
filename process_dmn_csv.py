#!/usr/bin/env python3
"""
Process and aggregate DMN CSV files for WHO CHW workflow system.
This script creates DMN module CSV files and aggregates them into a final CSV.
"""

import csv
import json
import os
from typing import List, Dict, Any
from pathlib import Path

def load_rules_data() -> List[Dict[str, Any]]:
    """Load the extracted rules data from JSON files."""
    rules_data = []
    
    # Load from WHO CHW rules
    chw_rules_path = "/workspace/who_chw_workflow/01_extracted_data/chw_rules_extracted.json"
    if os.path.exists(chw_rules_path):
        with open(chw_rules_path, 'r') as f:
            chw_rules = json.load(f)
            rules_data.extend(chw_rules)
    
    # Load from OpenAI extracted rules
    openai_rules_path = "/workspace/openai_output_v4/01_extracted_data/openai_extracted_rules.json"
    if os.path.exists(openai_rules_path):
        with open(openai_rules_path, 'r') as f:
            openai_rules = json.load(f)
            rules_data.extend(openai_rules)
    
    return rules_data

def categorize_rules_by_module(rules_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize rules into different DMN modules based on their characteristics."""
    modules = {
        'dmn_module_a': [],  # High priority danger signs (priority >= 95)
        'dmn_module_b': [],  # Hospital referrals (priority 80-94)
        'dmn_module_c': [],  # Clinic referrals (priority 60-79)
        'dmn_module_d': [],  # Home care (priority 40-59)
        'dmn_module_e': []   # Follow-up and other (priority < 40)
    }
    
    for rule in rules_data:
        priority = rule.get('then', {}).get('priority', 50)
        triage = rule.get('then', {}).get('propose_triage', 'home')
        
        if priority >= 95 or 'danger' in str(rule.get('then', {}).get('set_flags', [])):
            modules['dmn_module_a'].append(rule)
        elif triage == 'hospital' and priority >= 80:
            modules['dmn_module_b'].append(rule)
        elif triage == 'clinic' or (priority >= 60 and priority < 80):
            modules['dmn_module_c'].append(rule)
        elif triage == 'home' and priority >= 40:
            modules['dmn_module_d'].append(rule)
        else:
            modules['dmn_module_e'].append(rule)
    
    return modules

def convert_rule_to_csv_row(rule: Dict[str, Any]) -> Dict[str, str]:
    """Convert a rule dictionary to a CSV row format."""
    rule_id = rule.get('rule_id', 'UNKNOWN')
    
    # Extract conditions
    conditions = []
    when_clauses = rule.get('when', [])
    for clause in when_clauses:
        if 'sym' in clause:
            condition = f"{clause['sym']} = {clause.get('eq', True)}"
        elif 'obs' in clause:
            op = clause.get('op', 'eq')
            value = clause.get('value', '')
            condition = f"{clause['obs']} {op} {value}"
        else:
            condition = str(clause)
        conditions.append(condition)
    
    # Extract actions
    then_clause = rule.get('then', {})
    triage = then_clause.get('propose_triage', 'home')
    flags = ', '.join(then_clause.get('set_flags', []))
    reasons = ', '.join(then_clause.get('reasons', []))
    guideline_ref = then_clause.get('guideline_ref', '')
    priority = then_clause.get('priority', 50)
    
    return {
        'rule_id': rule_id,
        'conditions': ' AND '.join(conditions),
        'triage_decision': triage,
        'flags': flags,
        'reasons': reasons,
        'guideline_reference': guideline_ref,
        'priority': str(priority)
    }

def create_dmn_module_csv(module_name: str, rules: List[Dict[str, Any]], output_dir: str):
    """Create a CSV file for a specific DMN module."""
    csv_path = os.path.join(output_dir, f"{module_name}.csv")
    
    fieldnames = [
        'rule_id', 
        'conditions', 
        'triage_decision', 
        'flags', 
        'reasons', 
        'guideline_reference', 
        'priority'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for rule in rules:
            csv_row = convert_rule_to_csv_row(rule)
            writer.writerow(csv_row)
    
    print(f"Created {csv_path} with {len(rules)} rules")
    return csv_path

def aggregate_csv_files(csv_files: List[str], output_path: str):
    """Aggregate multiple CSV files into a single final CSV file."""
    all_rows = []
    fieldnames = None
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                
                for row in reader:
                    row['source_module'] = os.path.basename(csv_file).replace('.csv', '')
                    all_rows.append(row)
    
    # Sort by priority (descending) and then by rule_id
    all_rows.sort(key=lambda x: (-int(x.get('priority', 50)), x.get('rule_id', '')))
    
    # Add source_module to fieldnames
    if fieldnames and 'source_module' not in fieldnames:
        fieldnames = list(fieldnames) + ['source_module']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Created aggregated file {output_path} with {len(all_rows)} total rules")
    return output_path

def main():
    """Main function to process and aggregate DMN CSV files."""
    print("Processing DMN CSV files for WHO CHW workflow system...")
    
    # Load rules data
    print("Loading rules data...")
    rules_data = load_rules_data()
    print(f"Loaded {len(rules_data)} rules")
    
    # Categorize rules by module
    print("Categorizing rules by module...")
    modules = categorize_rules_by_module(rules_data)
    
    # Create output directory
    output_dir = "/workspace"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual module CSV files
    csv_files = []
    for module_name, module_rules in modules.items():
        if module_rules:  # Only create files for modules with rules
            csv_path = create_dmn_module_csv(module_name, module_rules, output_dir)
            csv_files.append(csv_path)
        else:
            print(f"No rules found for {module_name}")
    
    # Create aggregate final CSV
    if csv_files:
        aggregate_path = os.path.join(output_dir, "dmn_aggregate_final.csv")
        aggregate_csv_files(csv_files, aggregate_path)
        
        print(f"\nSummary:")
        print(f"- Created {len(csv_files)} module CSV files")
        print(f"- Created aggregate file: {aggregate_path}")
        print(f"- Total rules processed: {len(rules_data)}")
        
        # Show module breakdown
        for module_name, module_rules in modules.items():
            print(f"- {module_name}: {len(module_rules)} rules")
    else:
        print("No CSV files were created - no rules data found")

if __name__ == "__main__":
    main()