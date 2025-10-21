#!/usr/bin/env python3
"""
DMN CSV Processor and Aggregator

This script processes DMN module CSV files and aggregates them into a final CSV file.
It handles the configuration format:
<instance id="dmn_module_a" src="jr://file-csv/dmn_module_a.csv"/>
<instance id="dmn_module_b" src="jr://file-csv/dmn_module_b.csv"/>
...
<instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import sys


class DMNCSVProcessor:
    """Processes and aggregates DMN module CSV files."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.modules = {}
        self.aggregated_data = []
    
    def parse_config(self, config_xml: str) -> List[Dict[str, str]]:
        """Parse the XML configuration to extract module instances."""
        try:
            root = ET.fromstring(config_xml)
            instances = []
            
            for instance in root.findall('.//instance'):
                instance_id = instance.get('id')
                src = instance.get('src')
                if instance_id and src:
                    instances.append({
                        'id': instance_id,
                        'src': src
                    })
            
            return instances
        except ET.ParseError as e:
            print(f"Error parsing XML configuration: {e}")
            return []
    
    def load_csv_file(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load a CSV file and return its contents as a list of dictionaries."""
        try:
            full_path = self.base_dir / csv_path
            if not full_path.exists():
                print(f"Warning: CSV file not found: {full_path}")
                return []
            
            with open(full_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            return []
    
    def process_module(self, module_id: str, csv_path: str) -> Dict[str, Any]:
        """Process a single DMN module CSV file."""
        print(f"Processing module: {module_id}")
        
        # Remove the jr://file-csv/ prefix if present
        clean_path = csv_path.replace('jr://file-csv/', '')
        
        data = self.load_csv_file(clean_path)
        
        module_info = {
            'module_id': module_id,
            'csv_path': clean_path,
            'data': data,
            'row_count': len(data),
            'columns': list(data[0].keys()) if data else []
        }
        
        self.modules[module_id] = module_info
        print(f"  Loaded {module_info['row_count']} rows with columns: {module_info['columns']}")
        
        return module_info
    
    def aggregate_modules(self) -> List[Dict[str, Any]]:
        """Aggregate all module data into a single dataset."""
        print("\nAggregating modules...")
        
        # Collect all unique columns across modules
        all_columns = set()
        for module in self.modules.values():
            all_columns.update(module['columns'])
        
        # Add module_id column to track source
        all_columns.add('module_id')
        all_columns = sorted(list(all_columns))
        
        print(f"All columns: {all_columns}")
        
        # Aggregate data from all modules
        aggregated = []
        for module_id, module_info in self.modules.items():
            for row in module_info['data']:
                # Add module_id to each row
                row['module_id'] = module_id
                aggregated.append(row)
        
        self.aggregated_data = aggregated
        print(f"Aggregated {len(aggregated)} total rows from {len(self.modules)} modules")
        
        return aggregated
    
    def save_aggregated_csv(self, output_path: str) -> str:
        """Save the aggregated data to a CSV file."""
        if not self.aggregated_data:
            print("No data to save. Run aggregate_modules() first.")
            return ""
        
        # Handle both relative and absolute paths
        if Path(output_path).is_absolute():
            output_file = Path(output_path)
        else:
            output_file = self.base_dir / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all unique columns from aggregated data
        all_columns = set()
        for row in self.aggregated_data:
            all_columns.update(row.keys())
        all_columns = sorted(list(all_columns))
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(self.aggregated_data)
        
        print(f"Saved aggregated data to: {output_file}")
        return str(output_file)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the processing."""
        report = {
            'total_modules': len(self.modules),
            'total_rows': len(self.aggregated_data),
            'modules': {}
        }
        
        for module_id, module_info in self.modules.items():
            report['modules'][module_id] = {
                'csv_path': module_info['csv_path'],
                'row_count': module_info['row_count'],
                'columns': module_info['columns']
            }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Process and aggregate DMN CSV files")
    parser.add_argument("--config", help="XML configuration file path")
    parser.add_argument("--config-xml", help="XML configuration as string")
    parser.add_argument("--base-dir", default=".", help="Base directory for CSV files")
    parser.add_argument("--output", default="dmn_aggregate_final.csv", help="Output CSV file")
    parser.add_argument("--report", help="Save summary report to JSON file")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DMNCSVProcessor(args.base_dir)
    
    # Get configuration
    config_xml = ""
    if args.config:
        with open(args.config, 'r') as f:
            config_xml = f.read()
    elif args.config_xml:
        config_xml = args.config_xml
    else:
        # Use the configuration from the user's input
        config_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<instances>
    <instance id="dmn_module_a" src="jr://file-csv/dmn_module_a.csv"/>
    <instance id="dmn_module_b" src="jr://file-csv/dmn_module_b.csv"/>
    <instance id="dmn_module_c" src="jr://file-csv/dmn_module_c.csv"/>
    <instance id="dmn_module_d" src="jr://file-csv/dmn_module_d.csv"/>
    <instance id="dmn_module_e" src="jr://file-csv/dmn_module_e.csv"/>
    <instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
</instances>'''
    
    # Parse configuration
    instances = processor.parse_config(config_xml)
    if not instances:
        print("No instances found in configuration")
        sys.exit(1)
    
    print(f"Found {len(instances)} module instances")
    
    # Process each module (except the aggregate_final one)
    for instance in instances:
        if instance['id'] != 'dmn_aggregate_final':
            processor.process_module(instance['id'], instance['src'])
    
    # Aggregate all modules
    processor.aggregate_modules()
    
    # Save aggregated data
    output_file = processor.save_aggregated_csv(args.output)
    
    # Generate and save report
    report = processor.generate_summary_report()
    print(f"\nSummary Report:")
    print(f"  Total modules: {report['total_modules']}")
    print(f"  Total rows: {report['total_rows']}")
    
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {args.report}")
    
    print(f"\nProcessing complete! Output saved to: {output_file}")


if __name__ == "__main__":
    main()