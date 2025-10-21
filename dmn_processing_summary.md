# DMN CSV Processing Summary

## Overview
Successfully processed and aggregated DMN (Decision Model and Notation) CSV files for the WHO CHW (Community Health Worker) workflow system.

## Files Created

### Individual Module Files
- **dmn_module_a.csv** (134 rules) - High priority danger signs (priority ≥ 95)
  - Contains critical medical conditions requiring immediate hospital referral
  - Includes rules for convulsions, severe breathing issues, malnutrition, etc.

- **dmn_module_b.csv** (1 rule) - Hospital referrals (priority 80-94)
  - Contains moderate priority hospital referral rules

- **dmn_module_c.csv** (45 rules) - Clinic referrals (priority 60-79)
  - Contains rules for conditions requiring clinic-level care
  - Includes fever management, breathing issues, malnutrition monitoring

- **dmn_module_d.csv** (23 rules) - Home care (priority 40-59)
  - Contains rules for conditions that can be managed at home
  - Includes follow-up care and minor symptoms

- **dmn_module_e.csv** (5 rules) - Follow-up and other (priority < 40)
  - Contains miscellaneous rules and follow-up procedures

### Aggregate File
- **dmn_aggregate_final.csv** (208 rules total)
  - Combined all module rules into a single file
  - Sorted by priority (descending) and rule ID
  - Includes source module information for traceability

## Data Structure
Each CSV file contains the following columns:
- `rule_id`: Unique identifier for the rule
- `conditions`: Logical conditions that trigger the rule
- `triage_decision`: Recommended care level (hospital/clinic/home)
- `flags`: Associated flags (danger_sign, clinic_referral, etc.)
- `reasons`: Human-readable reasons for the decision
- `guideline_reference`: Reference to WHO-IMCI guidelines
- `priority`: Numerical priority (higher = more urgent)
- `source_module`: (aggregate file only) Source module name

## Rule Distribution
- **Total Rules Processed**: 208
- **High Priority (Module A)**: 134 rules (64.4%)
- **Hospital Referrals (Module B)**: 1 rule (0.5%)
- **Clinic Referrals (Module C)**: 45 rules (21.6%)
- **Home Care (Module D)**: 23 rules (11.1%)
- **Follow-up (Module E)**: 5 rules (2.4%)

## Source Data
Rules were extracted from:
- WHO CHW workflow rules (`/workspace/who_chw_workflow/01_extracted_data/chw_rules_extracted.json`)
- OpenAI extracted rules (`/workspace/openai_output_v4/01_extracted_data/openai_extracted_rules.json`)

## Usage
These CSV files can be used for:
- DMN model implementation
- Clinical decision support systems
- Training healthcare workers
- Quality assurance and validation
- Integration with electronic health records

## File Sizes
- dmn_aggregate_final.csv: 24.4 KB
- dmn_module_a.csv: 13.7 KB
- dmn_module_c.csv: 4.8 KB
- dmn_module_d.csv: 2.6 KB
- dmn_module_e.csv: 678 bytes
- dmn_module_b.csv: 157 bytes

Generated on: $(date)