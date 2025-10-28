# DMN CSV Integration Guide

## Overview

This guide explains how to integrate the generated DMN decision tables as modular CSV files into your XForms/ODK application.

## Generated Files

The following CSV files have been created in `forms/app/orchestrator-media/`:

- `dmn_module_a.csv` - Module A decision rules (8 rules, 467 bytes)
- `dmn_module_b.csv` - Module B decision rules (8 rules, 483 bytes)
- `dmn_module_c.csv` - Module C decision rules (5 rules, 294 bytes)
- `dmn_module_d.csv` - Module D decision rules (4 rules, 250 bytes)
- `dmn_module_e.csv` - Module E decision rules (4 rules, 248 bytes)
- `dmn_aggregate_final.csv` - Final aggregation logic (3 rules, 197 bytes)

## CSV Structure

Each CSV file has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `key` | string | Rule identifier (r001, r002, etc.) |
| `triage` | string | Triage decision: "hospital", "clinic", or "home" |
| `danger_sign` | boolean | true/false - indicates if danger sign present |
| `clinic_referral` | boolean | true/false - indicates if clinic referral needed |
| `reason` | string | Snake_case reason for the decision |
| `ref` | string | Page reference from guideline (e.g., "p6-6", "p24") |
| `advice` | string | Advice text (currently empty per policy) |

## XForm Integration

### 1. Instance Declarations

Add these instance declarations to your XForm's `<model>` section:

```xml
<instance id="dmn_module_a"        src="jr://file-csv/dmn_module_a.csv"/>
<instance id="dmn_module_b"        src="jr://file-csv/dmn_module_b.csv"/>
<instance id="dmn_module_c"        src="jr://file-csv/dmn_module_c.csv"/>
<instance id="dmn_module_d"        src="jr://file-csv/dmn_module_d.csv"/>
<instance id="dmn_module_e"        src="jr://file-csv/dmn_module_e.csv"/>
<instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
```

### 2. Calculating the Key

For each module, you need to calculate which rule to apply based on your input conditions. This is typically done with a series of `if()` statements:

```xml
<bind nodeset="/data/module_a_key" type="string" calculate="
  if(${condition1}, 'r001',
  if(${condition2}, 'r002',
  if(${condition3}, 'r003',
  ...
  '')))
"/>
```

### 3. Using pulldata()

Once you have the key, use `pulldata()` to retrieve values:

```xml
<!-- Get triage decision from module_a -->
<bind nodeset="/data/module_a_triage" type="string" 
      calculate="pulldata('dmn_module_a', 'triage', 'key', /data/module_a_key)"/>

<!-- Get danger sign from module_a -->
<bind nodeset="/data/module_a_danger_sign" type="string" 
      calculate="pulldata('dmn_module_a', 'danger_sign', 'key', /data/module_a_key)"/>

<!-- Get clinic referral from module_a -->
<bind nodeset="/data/module_a_clinic_referral" type="string" 
      calculate="pulldata('dmn_module_a', 'clinic_referral', 'key', /data/module_a_key)"/>

<!-- Get reason from module_a -->
<bind nodeset="/data/module_a_reason" type="string" 
      calculate="pulldata('dmn_module_a', 'reason', 'key', /data/module_a_key)"/>

<!-- Get reference from module_a -->
<bind nodeset="/data/module_a_ref" type="string" 
      calculate="pulldata('dmn_module_a', 'ref', 'key', /data/module_a_key)"/>
```

## Example: Module A Decision Logic

Module A handles danger sign assessment:

- **r001**: Any danger sign → Hospital (ref: p6-6)
- **r002**: Convulsions → Hospital (ref: p43)
- **r003**: Chest indrawing → Hospital (ref: p24)
- **r004**: Unusually sleepy/unconscious → Hospital (ref: p33)
- **r005**: Vomits everything → Hospital (ref: p41)
- **r006**: Cannot drink/feed → Hospital (ref: p50)
- **r007**: Severe malnutrition → Hospital (ref: p43)
- **r008**: No danger sign → Home (ref: p7)

## Example: Aggregate Final Logic

The aggregate module combines outputs from all modules:

- **r001**: If any module has danger_sign=true → Hospital (priority)
- **r002**: Else if any module has clinic_referral=true → Clinic
- **r003**: Else → Home (treat at home)

## File Deployment

### For ODK Collect / Enketo

1. Place all CSV files in your form's media folder
2. The CSV files will be bundled with the form when deployed
3. Files must be in the same directory as referenced by the `jr://file-csv/` URI

### For CHT (Community Health Toolkit)

1. Place CSV files in the appropriate media directory
2. Update your forms XML to include the instance declarations
3. Deploy using `cht --upload-app-forms`

## Troubleshooting

### pulldata() returns empty

- Verify CSV file is in the media folder
- Check that instance ID matches the pulldata() first parameter (without `.csv`)
- Ensure the key exists in the CSV file
- Verify column name is spelled correctly

### Boolean values not working

- CSV stores booleans as strings: "true" or "false"
- In XPath, compare with: `${module_a_danger_sign} = 'true'`
- Not: `${module_a_danger_sign} = true()`

### Page references need normalization

Page references have been automatically normalized:
- Ranges sorted: `p11-10` → `p10-11`
- Spaces removed: `P 23 - 7` → `p7-23`
- Prefix added: `15-17` → `p15-17`

## Source Files

The CSV files were generated from:
- **Source DMN**: `chatchw/chatchw/out/decisions.dmn`
- **Variables**: `chatchw/chatchw/out/merged_ir.json`
- **Generator**: `generate_dmn_csvs.py`

## Regeneration

To regenerate the CSV files after updating the DMN:

```bash
python3 generate_dmn_csvs.py
```

This will overwrite the existing CSV files in `forms/app/orchestrator-media/`.
