#!/bin/bash

# DMN Module CSV Processor
# This script handles the XML configuration and processes DMN CSV files

set -e

# Configuration
BASE_DIR="."
OUTPUT_CSV="dmn_aggregate_final.csv"
REPORT_JSON="dmn_processing_report.json"

# Create the XML configuration file from the provided instances
create_config() {
    cat > dmn_config.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<instances>
    <instance id="dmn_module_a" src="jr://file-csv/dmn_module_a.csv"/>
    <instance id="dmn_module_b" src="jr://file-csv/dmn_module_b.csv"/>
    <instance id="dmn_module_c" src="jr://file-csv/dmn_module_c.csv"/>
    <instance id="dmn_module_d" src="jr://file-csv/dmn_module_d.csv"/>
    <instance id="dmn_module_e" src="jr://file-csv/dmn_module_e.csv"/>
    <instance id="dmn_aggregate_final" src="jr://file-csv/dmn_aggregate_final.csv"/>
</instances>
EOF
    echo "Created dmn_config.xml"
}

# Process the DMN modules
process_modules() {
    echo "Processing DMN modules..."
    
    # Check if Python script exists
    if [ ! -f "process_dmn_csv.py" ]; then
        echo "Error: process_dmn_csv.py not found"
        exit 1
    fi
    
    # Run the Python processor
    python3 process_dmn_csv.py \
        --base-dir "$BASE_DIR" \
        --config dmn_config.xml \
        --output "$OUTPUT_CSV" \
        --report "$REPORT_JSON"
    
    echo "Processing complete!"
}

# Create sample CSV files for testing
create_samples() {
    echo "Creating sample CSV files..."
    python3 create_sample_dmn_csvs.py
}

# Main function
main() {
    case "${1:-process}" in
        "create-config")
            create_config
            ;;
        "create-samples")
            create_samples
            ;;
        "process")
            create_config
            process_modules
            ;;
        "full-test")
            create_config
            create_samples
            process_modules
            ;;
        *)
            echo "Usage: $0 {create-config|create-samples|process|full-test}"
            echo ""
            echo "Commands:"
            echo "  create-config  - Create the XML configuration file"
            echo "  create-samples - Create sample CSV files for testing"
            echo "  process        - Process existing CSV files (default)"
            echo "  full-test      - Create samples and process them"
            exit 1
            ;;
    esac
}

main "$@"