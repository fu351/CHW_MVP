# DMN CSV Module Processing

This directory contains DMN (Decision Model and Notation) module CSV files and processing scripts for clinical decision support systems.

## Files Overview

### DMN Module CSV Files
- `dmn_module_a.csv` - Convulsion assessment module
- `dmn_module_b.csv` - Fever assessment module  
- `dmn_module_c.csv` - Breathing assessment module
- `dmn_module_d.csv` - Dehydration assessment module
- `dmn_module_e.csv` - Skin assessment module

### Aggregate File
- `dmn_aggregate_final.csv` - Combined data from all modules

### Processing Scripts
- `process_dmn_csv.py` - Python script to process and aggregate CSV files
- `setup_dmn_instances.sh` - Bash script to validate and display DMN instances

## DMN Instance References

The DMN instances can be referenced using the following format:

```xml
<instance id="dmn_module_a"        src="jr://file-csv/dmn_module_a.csv"/>
<instance id="dmn_module_b"        src="jr://file-csv/dmn_module_b.csv"/>
<instance id="dmn_module_c"        src="jr://file-csv/dmn_module_c.csv"/>
<instance id="dmn_module_d"        src="jr://file-csv/dmn_module_d.csv"/>
<instance id="dmn_module_e"        src="jr://file-csv/dmn_module_e.csv"/>
<instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
```

## CSV File Structure

Each CSV file contains the following columns:
- `key` - Unique identifier for each rule (e.g., r001, r002)
- `triage` - Triage decision (home, clinic)
- `danger_sign` - Boolean indicating if danger signs are present
- `clinic_referral` - Boolean indicating if clinic referral is needed
- `reason` - Explanation for the decision
- `ref` - Page reference (e.g., p1, p2)
- `advice` - Clinical advice for the patient

## Usage

### Validate and Display Instances
```bash
./setup_dmn_instances.sh
```

### Process and Aggregate CSV Files
```bash
python3 process_dmn_csv.py
```

## Clinical Decision Logic

The CSV files implement clinical decision trees for Community Health Worker (CHW) assessments:

1. **Module A (Convulsion)**: Assesses for convulsion/seizure activity
2. **Module B (Fever)**: Evaluates fever severity and urgency
3. **Module C (Breathing)**: Checks for respiratory distress
4. **Module D (Dehydration)**: Assesses hydration status
5. **Module E (Skin)**: Evaluates skin conditions and infections

Each module follows a triage approach:
- **Home**: Continue monitoring at home
- **Clinic**: Refer to clinic for assessment
- **Emergency**: Immediate medical attention required

## Integration

These CSV files are designed to work with:
- ODK Collect forms (using `jr://file-csv/` references)
- Clinical decision support systems
- Mobile health applications
- Electronic health records

## File Validation

All CSV files have been validated for:
- Proper CSV format
- Required column headers
- Data consistency
- UTF-8 encoding