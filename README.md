# ChatCHW - Community Health Worker Decision Support System

[![Standards](https://img.shields.io/badge/BPMN-2.0-blue)](https://www.omg.org/spec/BPMN/)
[![Standards](https://img.shields.io/badge/DMN-1.3-green)](https://www.omg.org/spec/DMN/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen)](https://python.org)

ChatCHW is a **plug-and-play** clinical decision support system that transforms WHO community health guidelines into interactive chatbot workflows. It generates standards-compliant BPMN and DMN artifacts for seamless integration with external systems.

## 🚀 **Key Features**

- **📋 Data-Driven**: Converts JSON clinical rules into BPMN/DMN automatically
- **🤖 Interactive Chatbot**: Guides CHWs through clinical assessments step-by-step  
- **⚙️ Standards Compliant**: Generates BPMN 2.0 and DMN 1.3 XML artifacts
- **🔄 Plug-and-Play**: Works with any WHO guidelines without code changes
- **✅ Validated**: Passes industry-standard validators (bpmnlint, dmnlint)
- **🌍 Portable**: Supports any clinical domain (malaria, fever, malnutrition, etc.)

## 📁 **Repository Structure**

```
CHW_MVP/
├── chatchw/                    # Main Python package
│   ├── chatchw/               # Core modules
│   │   ├── bpmn_builder.py    # BPMN 2.0 generation
│   │   ├── dmn_builder.py     # DMN 1.3 generation  
│   │   ├── chatbot_engine.py  # Interactive chatbot
│   │   ├── cli.py             # Command-line interface
│   │   ├── rules/             # Clinical rule definitions
│   │   └── ...
│   └── tests/                 # Test suites
├── outputs/                   # Generated artifacts
│   ├── workflow_process.bpmn  # BPMN workflow
│   └── decision_logic.dmn     # DMN decision tables
├── WHO CHW guide 2012.pdf     # Source guidelines
└── README.md                  # This file
```

## 🛠️ **Installation**

### Prerequisites
- Python 3.8+
- Node.js (for validators)

### Install ChatCHW
```bash
cd chatchw
pip install -e .
```

### Install Validators (Optional)
```bash
npm install -g bpmnlint dmnlint
```

## 🍎 **macOS Onboarding & Setup**

Follow these steps on macOS (zsh/bash). Commands assume you are in the repository root.

### 1) Prerequisites

```bash
# Install Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Ensure brew is set up (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Python 3.11 and Graphviz
brew install python@3.11 graphviz

# (Optional) Node.js for validators
brew install node
npm install -g bpmnlint dmnlint
```

Verify Graphviz:

```bash
dot -V
```

### 2) Create a virtual environment

```bash
# From repo root
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # should show 3.11.x
```

### 3) Install ChatCHW (editable)

```bash
cd chatchw
pip install -e .
cd ..

# Confirm CLI is registered
chatchw version || python -m chatchw.chatchw.cli version
```

### 4) Run the end-to-end pipeline on macOS

Option A (one command, recommended):

```bash
chatchw pdf-workflow --pdf "./WHO CHW guide 2012.pdf" --module WHO_CHW --out-dir ./chw_workflow_output

chatchw validate-artifacts \
  --bpmn ./chw_workflow_output/02_process_models/chw_workflow_process.bpmn \
  --dmn  ./chw_workflow_output/02_process_models/chw_decision_logic.dmn

chatchw chat-dynamic \
  --bpmn ./chw_workflow_output/02_process_models/chw_workflow_process.bpmn \
  --dmn  ./chw_workflow_output/02_process_models/chw_decision_logic.dmn \
  --session dynamic_session.json
```

If the `chatchw` command isn’t found, prefix with Python:

```bash
python -m chatchw.chatchw.cli pdf-workflow --pdf "./WHO CHW guide 2012.pdf" --module WHO_CHW --out-dir ./chw_workflow_output
python -m chatchw.chatchw.cli validate-artifacts --bpmn ./chw_workflow_output/02_process_models/chw_workflow_process.bpmn --dmn ./chw_workflow_output/02_process_models/chw_decision_logic.dmn
python -m chatchw.chatchw.cli chat-dynamic --bpmn ./chw_workflow_output/02_process_models/chw_workflow_process.bpmn --dmn ./chw_workflow_output/02_process_models/chw_decision_logic.dmn --session dynamic_session.json
```

Option B (manual control):

```bash
# 1) Extract rules from PDF to JSON
mkdir -p ./rules
chatchw extract-pdf --pdf "./WHO CHW guide 2012.pdf" --module WHO_CHW --out ./rules/who_chw.json

# 2) Generate BPMN and DMN
mkdir -p ./outputs
chatchw generate-bpmn --rules ./rules --out ./outputs/who_chw_workflow.bpmn --format xml --check
chatchw generate-dmn  --rules ./rules --out ./outputs/who_chw_decisions.dmn   --format xml --check

# 3) Validate artifacts
chatchw validate-artifacts --bpmn ./outputs/who_chw_workflow.bpmn --dmn ./outputs/who_chw_decisions.dmn

# 4) Run the chatbot (dynamic engine)
chatchw chat-dynamic --bpmn ./outputs/who_chw_workflow.bpmn --dmn ./outputs/who_chw_decisions.dmn --session dynamic_session.json

# (Optional) Generate readable JSON and flowcharts
chatchw convert-to-json    --input ./outputs/who_chw_workflow.bpmn  --output ./outputs/bpmn_readable.json --type bpmn
chatchw convert-to-json    --input ./outputs/who_chw_decisions.dmn  --output ./outputs/dmn_readable.json  --type dmn
chatchw generate-flowchart --input ./outputs/who_chw_workflow.bpmn  --output ./outputs/bpmn_flow.png      --type bpmn
chatchw generate-flowchart --input ./outputs/who_chw_decisions.dmn  --output ./outputs/dmn_flow.png       --type dmn
```

Tips:
- On Apple Silicon, ensure Homebrew is under `/opt/homebrew` and your shell is initialized with `brew shellenv` as above.
- If Graphviz isn’t found, re-run `brew install graphviz` and restart your shell.
- For advanced AI-assisted extraction and modular artifacts, see `guarded-workflow` and `generate-modular-bpmn-dmn` in the CLI.

## 🎯 **Quick Start**

### 1. Generate BPMN and DMN from Rules
```bash
# Generate BPMN workflow
python -m chatchw.chatchw.cli generate-bpmn \
  --rules chatchw/rules \
  --out outputs/workflow.bpmn \
  --format xml \
  --check

# Generate DMN decision logic  
python -m chatchw.chatchw.cli generate-dmn \
  --rules chatchw/rules \
  --out outputs/decisions.dmn \
  --format xml \
  --check
```

### 2. Interactive Chatbot Session
```python
from chatchw.chatchw.chatbot_engine import ChatbotEngine

# Load your BPMN/DMN artifacts
engine = ChatbotEngine('outputs/workflow.bpmn', 'outputs/decisions.dmn')

# Start clinical assessment
state = engine.start_conversation()

# Answer questions interactively
question = engine.get_next_question(state)
print(f"Question: {question.text}")

# Process answer (example: convulsion detected)
state, valid = engine.process_answer(state, 'convulsion', True)

# Get final recommendation
recommendation = engine.get_final_recommendation(state)
print(f"Triage: {recommendation['outcome']}")  # "Hospital"
print(f"Reasoning: {recommendation['reasoning']}")
```

### 3. Validate Artifacts
```bash
# Validate with industry-standard tools
bpmnlint outputs/workflow.bpmn
dmnlint outputs/decisions.dmn

# Built-in validation
python -m chatchw.chatchw.cli validate-artifacts \
  --bpmn outputs/workflow.bpmn \
  --dmn outputs/decisions.dmn
```

## 🧭 **End-to-End Pipeline (PDF → Rules → BPMN/DMN → Chatbot)**

Below are two ways to run the full pipeline on Windows PowerShell.

### Option A: One-command pipeline (recommended)

```powershell
# From repo root, ensure package is installed (creates the `chatchw` command)
cd .\chatchw
pip install -e .
cd ..

# Run complete PDF → BPMN/DMN → JSON → Flowcharts pipeline
chatchw pdf-workflow --pdf ".\WHO CHW guide 2012.pdf" --module WHO_CHW --out-dir ".\chw_workflow_output"

# Validate the generated artifacts
chatchw validate-artifacts --bpmn ".\chw_workflow_output\02_process_models\chw_workflow_process.bpmn" --dmn ".\chw_workflow_output\02_process_models\chw_decision_logic.dmn"

# Start the interactive chatbot (dynamic engine)
chatchw chat-dynamic --bpmn ".\chw_workflow_output\02_process_models\chw_workflow_process.bpmn" --dmn ".\chw_workflow_output\02_process_models\chw_decision_logic.dmn" --session "dynamic_session.json"
```

If the `chatchw` command is not available, replace it with `python -m chatchw.chatchw.cli` in the commands above.

### Option B: Manual steps (more control)

```powershell
# 0) Install editable package once (from repo root)
cd .\chatchw
pip install -e .
cd ..

# 1) Extract JSON rules from the WHO PDF
New-Item -ItemType Directory -Force .\rules | Out-Null
chatchw extract-pdf --pdf ".\WHO CHW guide 2012.pdf" --module WHO_CHW --out ".\rules\who_chw.json"

# 2) Generate BPMN and DMN from the extracted rules
New-Item -ItemType Directory -Force .\outputs | Out-Null
chatchw generate-bpmn --rules ".\rules" --out ".\outputs\who_chw_workflow.bpmn" --format xml --check
chatchw generate-dmn  --rules ".\rules" --out ".\outputs\who_chw_decisions.dmn"   --format xml --check

# 3) Validate BPMN/DMN alignment (optional but recommended)
chatchw validate-artifacts --bpmn ".\outputs\who_chw_workflow.bpmn" --dmn ".\outputs\who_chw_decisions.dmn"

# 4) Run the chatbot (dynamic engine)
chatchw chat-dynamic --bpmn ".\outputs\who_chw_workflow.bpmn" --dmn ".\outputs\who_chw_decisions.dmn" --session "dynamic_session.json"

# (Optional) Generate readable JSON and flowcharts
chatchw convert-to-json      --input ".\outputs\who_chw_workflow.bpmn"  --output ".\outputs\bpmn_readable.json" --type bpmn
chatchw convert-to-json      --input ".\outputs\who_chw_decisions.dmn" --output ".\outputs\dmn_readable.json"  --type dmn
chatchw generate-flowchart   --input ".\outputs\who_chw_workflow.bpmn"  --output ".\outputs\bpmn_flow.png"     --type bpmn
chatchw generate-flowchart   --input ".\outputs\who_chw_decisions.dmn" --output ".\outputs\dmn_flow.png"      --type dmn
```

### Notes
- The dynamic chatbot engine (`chat-dynamic`) provides the most complete consultation flow and reasoning output.
- Use `chat-validate` to quickly sanity-check a pair of BPMN/DMN files for chatbot compatibility.
- For an AI-assisted extraction and modular artifacts, see the advanced `guarded-workflow` and `generate-modular-bpmn-dmn` commands in the CLI.

## 📋 **Clinical Rule Format**

Rules are defined in JSON format following this schema:

```json
{
  "fever": [
    {
      "rule_id": "FEV-99",
      "when": [
        {"sym": "convulsion", "eq": true}
      ],
      "then": {
        "propose_triage": "hospital",
        "set_flags": ["danger_sign"],
        "reasons": ["convulsion.danger.sign"],
        "priority": 100
      }
    }
  ]
}
```

### Rule Components
- **`when`**: Conditions (observations, symptoms, logic operators)
- **`then`**: Actions (triage decisions, flags, clinical reasoning)
- **`priority`**: Rule precedence (higher = evaluated first)

### Supported Conditions
```json
{"obs": "temp", "op": "ge", "value": 38.5}          // Temperature ≥ 38.5°C
{"sym": "convulsion", "eq": true}                   // Symptom present
{"any_of": [...]}                                   // OR logic
{"all_of": [...]}                                   // AND logic
```

## 🏥 **Clinical Workflow**

The system implements a **Hospital > Clinic > Home** triage hierarchy:

1. **🚨 Emergency (Hospital)**: Danger signs (convulsion, severe malnutrition)
2. **🏪 Moderate (Clinic)**: Concerning symptoms (dysentery, low MUAC)  
3. **🏠 Normal (Home)**: Routine care and monitoring

### Example Decision Flow
```
Patient presents → Assess danger signs → Convulsion detected
                                      ↓
                               Set danger_sign flag
                                      ↓
                              Route to Hospital triage
```

## ⚙️ **Standards Compliance**

### DMN 1.3 Features
- ✅ Namespace: `https://www.omg.org/spec/DMN/20191111/MODEL/`
- ✅ `<dmn:variable>` with FEEL `typeRef` (number/boolean/string)
- ✅ Nested `<dmn:text>` in expressions and entries
- ✅ `hitPolicy="FIRST"` for deterministic evaluation
- ✅ Conformance level declarations

### BPMN 2.0 Features  
- ✅ Default sequence flows on exclusive gateways
- ✅ `xsi:type="tFormalExpression"` on condition expressions
- ✅ Proper `<incoming>/<outgoing>` connectivity
- ✅ Parallel gateway splits (no start event fan-out)

### FEEL Subset Support
```
// Supported FEEL expressions:
temp >= 38.5              // Numeric comparisons
convulsion = true          // Boolean conditions  
temp < 37.0               // Range checks
```

## 🧪 **Testing**

### Run Test Suite
```bash
cd chatchw
python -m pytest tests/ -v
```

### Test Categories
- **Standards Compliance**: DMN 1.3, BPMN 2.0 validation
- **Clinical Logic**: Emergency detection, triage routing
- **Engine Functionality**: Question flow, decision evaluation
- **CLI Features**: Generation, validation, format options

### Manual Testing
```python
# Test emergency case
engine = ChatbotEngine('outputs/workflow.bpmn', 'outputs/decisions.dmn')
state = engine.start_conversation()
state, _ = engine.process_answer(state, 'convulsion', True)
rec = engine.get_final_recommendation(state)
assert rec['outcome'] == 'Hospital'  # Should route to hospital
```

## 🔧 **CLI Reference**

### Generation Commands
```bash
# BPMN Generation
python -m chatchw.chatchw.cli generate-bpmn --rules DIR --out FILE [--format xml|json] [--check]

# DMN Generation  
python -m chatchw.chatchw.cli generate-dmn --rules DIR --out FILE [--format xml|json] [--check]

# Combined Workflow
python -m chatchw.chatchw.cli pdf-to-bpmn --pdf FILE --out-dir DIR
python -m chatchw.chatchw.cli pdf-to-dmn --pdf FILE --out-dir DIR
```

### Validation Commands
```bash
# Validate Artifacts
python -m chatchw.chatchw.cli validate-artifacts --bpmn FILE --dmn FILE

# Check Alignment
python -m chatchw.chatchw.cli chat-validate --bpmn FILE --dmn FILE
```

### Extraction Commands
```bash
# OpenAI-Powered Extraction
python -m chatchw.chatchw.cli pdf-openai-workflow --pdf FILE --out-dir DIR

# Text Processing
python -m chatchw.chatchw.cli text-workflow --text "clinical guidelines..." --out-dir DIR
```

## 🚀 **Deployment**

### For New WHO Guidelines

1. **Extract Rules**:
   ```bash
   python -m chatchw.chatchw.cli pdf-openai-workflow \
     --pdf new_who_guidelines.pdf \
     --out-dir new_deployment/
   ```

2. **Generate Artifacts**:
   ```bash
   python -m chatchw.chatchw.cli generate-bpmn \
     --rules new_deployment/01_extracted_data/ \
     --out new_deployment/workflow.bpmn
   ```

3. **Deploy Chatbot**:
   ```python
   engine = ChatbotEngine('new_deployment/workflow.bpmn', 'new_deployment/decisions.dmn')
   # Ready to use - no code changes needed!
   ```

### Integration Examples

**Camunda Integration**:
```bash
# Deploy to Camunda BPM
curl -X POST "http://camunda:8080/engine-rest/deployment/create" \
  -F "deployment-name=CHW-Workflow" \
  -F "workflow.bpmn=@outputs/workflow.bpmn"
```

**Decision Service**:
```python
# Use as REST API
from flask import Flask, request
app = Flask(__name__)
engine = ChatbotEngine('workflow.bpmn', 'decisions.dmn')

@app.route('/assess', methods=['POST'])
def assess_patient():
    symptoms = request.json
    state = engine.start_conversation()
    for var, value in symptoms.items():
        state, _ = engine.process_answer(state, var, value)
    return engine.get_final_recommendation(state)
```

## 🤝 **Contributing**

### Adding New Medical Domains

1. **Create Rule Files**:
   ```json
   // malaria_rules.json
   {
     "malaria": [
       {
         "rule_id": "MAL-001",
         "when": [{"obs": "temp", "op": "ge", "value": 39.0}],
         "then": {"propose_triage": "clinic", "priority": 20}
       }
     ]
   }
   ```

2. **Generate Artifacts**:
   ```bash
   python -m chatchw.chatchw.cli generate-bpmn --rules malaria_rules/ --out malaria.bpmn
   ```

3. **Test Integration**:
   ```python
   engine = ChatbotEngine('malaria.bpmn', 'malaria.dmn')
   # System automatically adapts to malaria domain
   ```

### Development Setup
```bash
git clone https://github.com/your-org/CHW_MVP.git
cd CHW_MVP/chatchw
pip install -e ".[dev]"
pre-commit install
```

## 📚 **Documentation**

- **Standards**: [BPMN 2.0 Spec](https://www.omg.org/spec/BPMN/), [DMN 1.3 Spec](https://www.omg.org/spec/DMN/)
- **Clinical Guidelines**: WHO IMCI Guidelines 2014
- **FEEL Reference**: [DMN FEEL Spec](https://www.omg.org/spec/DMN/20191111/feel/)

## 🐛 **Known Limitations**

- **FEEL Subset**: Supports basic comparisons, not full FEEL specification
- **Rule Complexity**: Complex nested logic may require multiple rules
- **Validation**: Custom validator for clinical logic alignment

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgments**

- WHO for community health guidelines
- OMG for BPMN/DMN standards  
- Clinical domain experts for validation

---

**🎯 Ready to transform your medical guidelines into interactive decision support? Get started with ChatCHW today!**
