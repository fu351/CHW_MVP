#!/usr/bin/env python3
"""
Script to process and aggregate DMN CSV files.
This script reads individual DMN module CSV files and creates an aggregated output.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return its contents as a list of dictionaries."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return []
    
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def process_dmn_modules(workspace_dir: str = "/workspace") -> None:
    """Process all DMN module CSV files and create aggregated output."""
    
    # Define the module files
    module_files = [
        "dmn_module_a.csv",
        "dmn_module_b.csv", 
        "dmn_module_c.csv",
        "dmn_module_d.csv",
        "dmn_module_e.csv"
    ]
    
    # Read all module data
    all_data = []
    module_names = ['a', 'b', 'c', 'd', 'e']
    
    for i, module_file in enumerate(module_files):
        file_path = os.path.join(workspace_dir, module_file)
        module_data = read_csv_file(file_path)
        
        # Add module identifier to each row
        for row in module_data:
            row['module'] = module_names[i]
            row['decision_id'] = f"module_{module_names[i]}_decision"
            all_data.append(row)
    
    # Write aggregated data
    output_file = os.path.join(workspace_dir, "dmn_aggregate_final.csv")
    
    if all_data:
        # Get all unique fieldnames from all rows
        fieldnames = set()
        for row in all_data:
            fieldnames.update(row.keys())
        
        # Ensure consistent order
        ordered_fieldnames = ['key', 'module', 'decision_id', 'triage', 'danger_sign', 
                            'clinic_referral', 'reason', 'ref', 'advice']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Aggregated data written to {output_file}")
        print(f"✓ Processed {len(all_data)} rows from {len(module_files)} modules")
    else:
        print("❌ No data found to process")

def validate_csv_files(workspace_dir: str = "/workspace") -> bool:
    """Validate that all required CSV files exist and are readable."""
    module_files = [
        "dmn_module_a.csv",
        "dmn_module_b.csv", 
        "dmn_module_c.csv",
        "dmn_module_d.csv",
        "dmn_module_e.csv"
    ]
    
    all_valid = True
    for module_file in module_files:
        file_path = os.path.join(workspace_dir, module_file)
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {module_file}")
            all_valid = False
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv.Sniffer().sniff(f.read(1024))
                print(f"✓ Valid CSV: {module_file}")
            except Exception as e:
                print(f"❌ Invalid CSV {module_file}: {e}")
                all_valid = False
    
    return all_valid

def main():
    """Main function to process DMN CSV files."""
    workspace_dir = "/workspace"
    
    print("Processing DMN CSV files...")
    print("=" * 50)
    
    # Validate files first
    if not validate_csv_files(workspace_dir):
        print("\n❌ Some CSV files are missing or invalid")
        sys.exit(1)
    
    # Process the files
    process_dmn_modules(workspace_dir)
    
    print("\n✓ DMN CSV processing completed successfully!")

if __name__ == "__main__":
    main()