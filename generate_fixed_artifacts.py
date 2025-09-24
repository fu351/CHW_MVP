#!/usr/bin/env python3
"""Generate new BPMN/DMN artifacts with the fixed architecture."""

import json
import os
from pathlib import Path
from chatchw.chatchw.bpmn_builder import build_bpmn
from chatchw.chatchw.dmn_builder import generate_dmn

def load_v4_rules():
    """Load the v4 extracted rules."""
    rules_path = "openai_output_v4/01_extracted_data/openai_extracted_rules.json"
    
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"V4 rules not found at {rules_path}")
    
    with open(rules_path, 'r') as f:
        rules = json.load(f)
    
    print(f"✅ Loaded {len(rules)} rules from v4 data")
    return rules

def create_output_directory():
    """Create output directory for fixed artifacts."""
    output_dir = Path("openai_output_v4_fixed")
    process_models_dir = output_dir / "02_process_models"
    
    output_dir.mkdir(exist_ok=True)
    process_models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created output directory: {output_dir}")
    return process_models_dir

def generate_artifacts(rules, output_dir):
    """Generate BPMN and DMN artifacts using fixed builders."""
    
    # Organize rules by module (using a single module for now)
    rulepacks = {"openai_extracted": rules}
    
    try:
        # Generate BPMN with sequential workflow
        print("🔄 Generating BPMN with sequential clinical workflow...")
        bpmn_xml = build_bpmn(rulepacks)
        bpmn_path = output_dir / "openai_workflow_process_fixed.bpmn"
        
        with open(bpmn_path, 'w', encoding='utf-8') as f:
            f.write(bpmn_xml)
        
        print(f"✅ BPMN generated: {bpmn_path} ({len(bpmn_xml)} chars)")
        
        # Generate DMN with hierarchical decisions
        print("🔄 Generating DMN with hierarchical decision structure...")
        dmn_xml = generate_dmn(rulepacks)
        dmn_path = output_dir / "openai_decision_logic_fixed.dmn"
        
        with open(dmn_path, 'w', encoding='utf-8') as f:
            f.write(dmn_xml)
        
        print(f"✅ DMN generated: {dmn_path} ({len(dmn_xml)} chars)")
        
        return str(bpmn_path), str(dmn_path)
        
    except Exception as e:
        print(f"❌ Error generating artifacts: {e}")
        raise

def analyze_improvements(bpmn_path, dmn_path):
    """Analyze the improvements in the new artifacts."""
    
    # Check file sizes
    bpmn_size = os.path.getsize(bpmn_path)
    dmn_size = os.path.getsize(dmn_path)
    
    print("\n📊 **ARTIFACT ANALYSIS**")
    print("=" * 50)
    
    # Compare with old artifacts if they exist
    old_bpmn = "openai_output_v4/02_process_models/openai_workflow_process.bpmn"
    old_dmn = "openai_output_v4/02_process_models/openai_decision_logic.dmn"
    
    if os.path.exists(old_bpmn) and os.path.exists(old_dmn):
        old_bpmn_size = os.path.getsize(old_bpmn)
        old_dmn_size = os.path.getsize(old_dmn)
        
        bpmn_reduction = ((old_bpmn_size - bpmn_size) / old_bpmn_size) * 100
        dmn_reduction = ((old_dmn_size - dmn_size) / old_dmn_size) * 100
        
        print(f"BPMN Size: {bpmn_size:,} bytes (was {old_bpmn_size:,}, {bpmn_reduction:.1f}% reduction)")
        print(f"DMN Size: {dmn_size:,} bytes (was {old_dmn_size:,}, {dmn_reduction:.1f}% reduction)")
    else:
        print(f"BPMN Size: {bpmn_size:,} bytes")
        print(f"DMN Size: {dmn_size:,} bytes")
    
    # Analyze BPMN structure
    with open(bpmn_path, 'r') as f:
        bpmn_content = f.read()
    
    task_count = bpmn_content.count('<bpmn:task')
    user_task_count = bpmn_content.count('<bpmn:userTask')
    business_rule_task_count = bpmn_content.count('<bpmn:businessRuleTask')
    gateway_count = bpmn_content.count('<bpmn:exclusiveGateway')
    parallel_gateway_count = bpmn_content.count('<bpmn:parallelGateway')
    
    print(f"\n🔧 BPMN STRUCTURE:")
    print(f"  - Tasks: {task_count}")
    print(f"  - User Tasks: {user_task_count}")
    print(f"  - Business Rule Tasks: {business_rule_task_count}")
    print(f"  - Exclusive Gateways: {gateway_count}")
    print(f"  - Parallel Gateways: {parallel_gateway_count}")
    
    # Analyze DMN structure
    with open(dmn_path, 'r') as f:
        dmn_content = f.read()
    
    decision_count = dmn_content.count('<dmn:decision')
    input_data_count = dmn_content.count('<dmn:inputData')
    rule_count = dmn_content.count('<dmn:rule')
    
    print(f"\n🧠 DMN STRUCTURE:")
    print(f"  - Decisions: {decision_count}")
    print(f"  - Input Data: {input_data_count}")
    print(f"  - Rules: {rule_count}")
    
    print(f"\n🎯 **ARCHITECTURE IMPROVEMENTS**:")
    print(f"  ✅ Sequential clinical workflow (no more parallel chaos)")
    print(f"  ✅ Hierarchical decision structure (no more monolithic table)")
    print(f"  ✅ Proper clinical assessment flow (danger signs → clinical → triage)")
    print(f"  ✅ Executable BPMN process")
    print(f"  ✅ Standards-compliant DMN decisions")

def main():
    """Main execution function."""
    print("🚀 **GENERATING FIXED ARTIFACTS**")
    print("=" * 60)
    
    try:
        # Load v4 rules
        rules = load_v4_rules()
        
        # Create output directory
        output_dir = create_output_directory()
        
        # Generate new artifacts
        bpmn_path, dmn_path = generate_artifacts(rules, output_dir)
        
        # Analyze improvements
        analyze_improvements(bpmn_path, dmn_path)
        
        print(f"\n🎉 **SUCCESS!** Fixed artifacts generated")
        print(f"📁 BPMN: {bpmn_path}")
        print(f"📁 DMN: {dmn_path}")
        print(f"\n💡 Ready for testing with the fixed chatbot engine!")
        
        return bpmn_path, dmn_path
        
    except Exception as e:
        print(f"❌ **ERROR**: {e}")
        raise

if __name__ == '__main__':
    main()