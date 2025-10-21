#!/bin/bash
# Script to set up DMN module instances and handle CSV file references
# This script creates the proper structure for DMN module CSV files

set -e  # Exit on any error

WORKSPACE_DIR="/workspace"
echo "Setting up DMN module instances..."
echo "=================================="

# Check if CSV files exist
CSV_FILES=(
    "dmn_module_a.csv"
    "dmn_module_b.csv" 
    "dmn_module_c.csv"
    "dmn_module_d.csv"
    "dmn_module_e.csv"
    "dmn_aggregate_final.csv"
)

echo "Checking for required CSV files:"
for file in "${CSV_FILES[@]}"; do
    if [ -f "$WORKSPACE_DIR/$file" ]; then
        echo "✓ Found: $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Create a summary of the DMN instances
echo ""
echo "DMN Module Instances:"
echo "===================="
echo "<instance id=\"dmn_module_a\"        src=\"jr://file-csv/dmn_module_a.csv\"/>"
echo "<instance id=\"dmn_module_b\"        src=\"jr://file-csv/dmn_module_b.csv\"/>"
echo "<instance id=\"dmn_module_c\"        src=\"jr://file-csv/dmn_module_c.csv\"/>"
echo "<instance id=\"dmn_module_d\"        src=\"jr://file-csv/dmn_module_d.csv\"/>"
echo "<instance id=\"dmn_module_e\"        src=\"jr://file-csv/dmn_module_e.csv\"/>"
echo "<instance id=\"dmn_aggregate_final\" src=\"jr://file-csv/dmn_aggregate_final.csv\"/>"

# Show file sizes
echo ""
echo "File sizes:"
for file in "${CSV_FILES[@]}"; do
    size=$(wc -c < "$WORKSPACE_DIR/$file")
    lines=$(wc -l < "$WORKSPACE_DIR/$file")
    echo "  $file: $size bytes, $lines lines"
done

echo ""
echo "✓ DMN module instances setup completed successfully!"