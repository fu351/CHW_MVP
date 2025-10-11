# ChatCHW v0.1 (Offline CLI MVP)

ChatCHW is a small, offline CLI that evaluates deterministic rules for community health workflows and generates intermediary BPMN and DMN artifacts. It renders BPMN via pm4py and a DMN Decision Requirements Diagram (DRD) via graphviz.
### Standards and Interop

- DMN: Emits DMN 1.3 namespace (`https://www.omg.org/spec/DMN/20191111/MODEL/`), includes `dmn:variable` with FEEL `typeRef` for each `inputData` (number/boolean/string), uses nested `dmn:text` for `inputExpression`, `inputEntry`, and `outputEntry`, and sets `hitPolicy="UNIQUE"` for deterministic tables.
- BPMN: Marks default sequence flows on `exclusiveGateway` via the `default` attribute, adds `xsi:type="tFormalExpression"` on all `bpmn:conditionExpression`, and avoids fanning out directly from the start event (uses a parallel split gateway).
- CLI: `generate-bpmn` and `generate-dmn` support `--format xml|json` and `--check` to validate outputs.

#### FEEL subset supported

- Comparators: `<, <=, >, >=, =`
- Booleans: `true/false` and simple equality tests
- List and interval unary tests are parsed in validators for overlap checks but generation focuses on simple expressions. Full FEEL spec is not implemented.

Known limits:
- No full FEEL evaluation engine; conditions are simple comparators/booleans.
- Unique hit policy enforced at generation; complex policies not modeled.
- DMN TCK compatibility is partial; a small subset is targeted by tests.


## Requirements

- Python 3.11
- Graphviz system binary installed and on PATH (for DMN DRD and some BPMN visualizations)

## Install

```bash
pip install -e .
```

Verify Graphviz is installed:
```bash
dot -V
```

## CLI Overview

### Interactive Medical Chatbot

**NEW**: ChatCHW now includes an interactive chatbot that uses BPMN/DMN files as conversation logic:

- **Interactive consultation** (recommended):
```bash
chatchw chat --bpmn workflow.bpmn --dmn decisions.dmn
```

- **Batch processing** with existing patient data:
```bash
chatchw chat-batch --bpmn workflow.bpmn --dmn decisions.dmn --input patient_data.json --output results.json
```

- **Validate BPMN/DMN** files for chatbot compatibility:
```bash
chatchw chat-validate --bpmn workflow.bpmn --dmn decisions.dmn
```

#### Example: Using Pipeline Outputs with Chatbot

```bash
# 1. Generate BPMN/DMN from WHO CHW PDF
chatchw pdf-workflow --pdf who_chw_guidelines.pdf --module malaria --out-dir malaria_workflow/

# 2. Start interactive chatbot with generated artifacts
chatchw chat --bpmn malaria_workflow/02_process_models/chw_workflow_process.bpmn --dmn malaria_workflow/02_process_models/chw_decision_logic.dmn

# 3. Or run batch processing
chatchw chat-batch --bpmn malaria_workflow/02_process_models/chw_workflow_process.bpmn --dmn malaria_workflow/02_process_models/chw_decision_logic.dmn --input patient_data.json
```

### Basic Workflow

- Initialize example input:
```bash
chatchw init --out input.encounter.example.json
```

- Run a decision on an input using bundled rules:
```bash
chatchw decide --input input.encounter.example.json --rules "$(python -c 'import pathlib,chatchw;print(pathlib.Path(chatchw.__file__).parent/\"rules\")')" --out decision.json
```

- Generate BPMN and render:
```bash
chatchw generate-bpmn --rules "$(python -c 'import pathlib,chatchw;print(pathlib.Path(chatchw.__file__).parent/\"rules\")')" --out model.bpmn
chatchw render-bpmn --bpmn model.bpmn --out model.svg
```

- Generate DMN and render the DRD:
```bash
chatchw generate-dmn --rules "$(python -c 'import pathlib,chatchw;print(pathlib.Path(chatchw.__file__).parent/\"rules\")')" --out model.dmn
chatchw render-dmn --dmn model.dmn --out model_drd.svg
```

- Export decisions to CSV from a JSONL log:
```bash
chatchw export-csv --logs encounters.jsonl --out encounters.csv
```

### PDF-to-Clinical-Workflow Pipeline

**NEW**: Transform WHO CHW PDF documents into comprehensive, organized clinical workflow outputs:

- **Complete workflow** (recommended):
```bash
chatchw pdf-workflow --pdf who_chw_guidelines.pdf --module malaria --out-dir chw_workflow_output/
```

#### Organized Output Structure

This command creates a well-organized file structure:

```
chw_workflow_output/
├── 01_extracted_data/
│   └── chw_rules_extracted.json          # Raw rules from PDF
├── 02_process_models/
│   ├── chw_workflow_process.bpmn         # BPMN process model
│   └── chw_decision_logic.dmn            # DMN decision model
├── 03_readable_formats/
│   ├── workflow_process_readable.json    # Human-readable BPMN
│   └── decision_logic_readable.json      # Human-readable DMN
├── 04_clinical_flowcharts/
│   └── chw_clinical_workflow_guide.png   # ⭐ MAIN USER GUIDE
└── 05_technical_diagrams/
    ├── technical_bpmn_flowchart.png       # Technical process flow
    ├── technical_dmn_flowchart.png        # Technical decision flow
    ├── bpmn_process_diagram.svg           # Standard BPMN diagram
    └── dmn_decision_diagram.svg           # Standard DMN diagram
```

#### Clinical Workflow Guide

The **key output** is `04_clinical_flowcharts/chw_clinical_workflow_guide.png` - a comprehensive visual guide that **replaces the original PDF** for field use:

- 📋 **Step-by-step decision logic** with clear IF/THEN conditions
- 🎨 **Color-coded priority levels**:
  - 🔴 **Red**: High priority/danger signs (≥100) - immediate action
  - 🟡 **Yellow**: Medium priority (50-99) - important findings  
  - 🟢 **Green**: Routine priority (<50) - standard assessments
- 📖 **Plain language conditions** (e.g., "temp ≥ 38.5", "Has convulsion")
- 🏥 **Clear referral paths** (Hospital/Clinic/Home)
- 💡 **Clinical reasoning** for each decision

- **Step-by-step approach**:
```bash
# Extract rules from PDF
chatchw extract-pdf --pdf who_chw_guidelines.pdf --module malaria --out malaria_rules.json

# Generate BPMN directly from PDF
chatchw pdf-to-bpmn --pdf who_chw_guidelines.pdf --module malaria --out malaria_model.bpmn

# Generate DMN directly from PDF  
chatchw pdf-to-dmn --pdf who_chw_guidelines.pdf --module malaria --out malaria_model.dmn

# Then render as usual
chatchw render-bpmn --bpmn malaria_model.bpmn --out malaria_bpmn.svg
chatchw render-dmn --dmn malaria_model.dmn --out malaria_drd.svg
```

### Flowchart Generation

Generate various types of visual workflows:

- **Clinical Flowcharts** (recommended for field use):
```bash
# Generate comprehensive clinical workflow from rules
chatchw generate-clinical-flowchart --rules fever_rules.json --module "Fever Management" --output clinical_guide.png
```

- **Technical Flowcharts** from BPMN/DMN:
```bash
# Generate technical flowchart from BPMN
chatchw generate-flowchart --input model.bpmn --output bpmn_flowchart.png --type bpmn

# Generate technical flowchart from DMN  
chatchw generate-flowchart --input model.dmn --output dmn_flowchart.png --type dmn
```

- **Convert to readable JSON format**:
```bash
# Convert BPMN to readable JSON
chatchw convert-to-json --input model.bpmn --output model_bpmn.json --type bpmn

# Convert DMN to readable JSON
chatchw convert-to-json --input model.dmn --output model_dmn.json --type dmn
```

### Utility Commands

- Version:
```bash
chatchw version
```

## PDF Processing Details

The PDF extraction feature automatically identifies:

- **Temperature conditions**: `temperature >= 38.5°C`
- **MUAC conditions**: `MUAC < 115 mm`
- **Symptom conditions**: convulsion, blood in stool, edema, etc.
- **Danger signs**: convulsion, seizure, severe dehydration, etc.
- **Referral actions**: hospital, clinic, home treatment
- **If-then patterns**: Various conditional statements in clinical text

The extractor uses pattern matching to convert natural language clinical guidelines into structured JSON rules that can be processed by the decision engine and transformed into BPMN/DMN artifacts.

## Notes

- This is a proof of concept. It performs no network calls and operates fully offline.
- Rules are stored as JSON, transformed into BPMN and DMN for transparency.
- PDF processing requires `pypdf` dependency and works best with well-structured WHO CHW documents.
- Rule extraction uses heuristic pattern matching and may require manual review for complex guidelines.
