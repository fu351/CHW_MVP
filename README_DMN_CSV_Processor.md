# DMN CSV Processor and Aggregator

This solution handles the XML configuration format you encountered and processes DMN (Decision Model and Notation) CSV files for aggregation.

## Problem Solved

The original error occurred because you were trying to run XML configuration data directly in bash:
```bash
<instance id="dmn_module_a"        src="jr://file-csv/dmn_module_a.csv"/>
<instance id="dmn_module_b"        src="jr://file-csv/dmn_module_b.csv"/>
# ... more instances
```

This caused bash syntax errors because XML is not valid bash syntax.

## Solution Components

### 1. `process_dmn_csv.py` - Main Processor
- Parses XML configuration to extract module instances
- Loads individual DMN module CSV files
- Aggregates all modules into a single CSV file
- Generates processing reports

### 2. `create_sample_dmn_csvs.py` - Sample Data Generator
- Creates sample DMN module CSV files for testing
- Generates XML configuration file
- Useful for testing and development

### 3. `process_dmn_modules.sh` - Shell Script Wrapper
- Provides convenient command-line interface
- Handles configuration file creation
- Supports different processing modes

## Usage

### Quick Start
```bash
# Create sample data and process it
./process_dmn_modules.sh full-test

# Or process existing CSV files
./process_dmn_modules.sh process
```

### Manual Processing
```bash
# Create sample CSV files
python3 create_sample_dmn_csvs.py

# Process the modules
python3 process_dmn_csv.py \
    --base-dir sample_dmn_csvs \
    --config sample_dmn_csvs/dmn_config.xml \
    --output dmn_aggregate_final.csv \
    --report processing_report.json
```

### Command Line Options
```bash
python3 process_dmn_csv.py --help
```

## Input Format

The processor expects XML configuration in this format:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<instances>
    <instance id="dmn_module_a" src="jr://file-csv/dmn_module_a.csv"/>
    <instance id="dmn_module_b" src="jr://file-csv/dmn_module_b.csv"/>
    <instance id="dmn_module_c" src="jr://file-csv/dmn_module_c.csv"/>
    <instance id="dmn_module_d" src="jr://file-csv/dmn_module_d.csv"/>
    <instance id="dmn_module_e" src="jr://file-csv/dmn_module_e.csv"/>
    <instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
</instances>
```

## Output

The processor generates:
- **Aggregated CSV**: All module data combined with `module_id` column
- **Processing Report**: JSON summary of processing results
- **Console Output**: Detailed processing information

## Sample Output

The aggregated CSV includes all rules from all modules:
```csv
action,condition,module_id,priority,rule_id
refer_to_geriatric,age >= 65,dmn_module_a,100,A001
check_fever,temperature > 38.5,dmn_module_a,50,A002
emergency_cardiac,chest_pain = true,dmn_module_b,100,B001
...
```

## Integration with ChatCHW

This processor integrates with your existing ChatCHW DMN pipeline:
- Handles the `jr://file-csv/` URI format used by ODK Collect
- Processes multiple DMN modules as defined in your configuration
- Generates aggregated output compatible with your workflow

## Files Created

- `process_dmn_csv.py` - Main processor script
- `create_sample_dmn_csvs.py` - Sample data generator
- `process_dmn_modules.sh` - Shell wrapper script
- `README_DMN_CSV_Processor.md` - This documentation

## Testing

The solution has been tested with:
- Sample DMN module CSV files (5 modules, 15 total rules)
- XML configuration parsing
- CSV aggregation and output generation
- Error handling for missing files

## Next Steps

1. Replace sample CSV files with your actual DMN module data
2. Update the XML configuration to match your specific modules
3. Integrate with your existing ChatCHW pipeline
4. Customize the aggregation logic as needed for your use case