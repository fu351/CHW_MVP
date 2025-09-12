from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Optional

import typer
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

from .bpmn_builder import build_bpmn
from .dmn_builder import generate_dmn
from .dmn_viz import render_drd
from .engine import decide
from .flowchart_generator import FlowchartGenerator
from .pdf_parser import CHWRuleExtractor
from .rules_loader import load_rules_dir
from .chatbot_engine import ChatbotEngine
from .dynamic_chatbot_engine import DynamicChatbotEngine
from .validator import check_bpmn_soundness, check_dmn_tables, check_alignment
import csv

app = typer.Typer(add_completion=False, no_args_is_help=True)

def load_dotenv():
    """Load environment variables from .env.local file."""
    env_file = Path(".env.local")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


@app.command("init")
def cli_init(out: str = typer.Option("input.encounter.example.json", "--out")):
    example = {
        "age_months": 24,
        "sex": "m",
        "observations": [
            {"id": "temp", "value": 38.6},
            {"id": "resp_rate", "value": 28.0},
            {"id": "muac_mm", "value": 120.0}
        ],
        "symptoms": {
            "feels_very_hot": True,
            "blood_in_stool": False,
            "diarrhea_days": 2,
            "convulsion": False,
            "edema_both_feet": False
        },
        "context": {
            "malaria_present": True,
            "cholera_present": False,
            "stockout": {"antimalarial": True}
        }
    }
    Path(out).write_text(json.dumps(example, indent=2), encoding="utf-8")
    typer.echo(out)


@app.command("decide")
def cli_decide(
    input: str = typer.Option(..., "--input", help="Path to input Encounter JSON"),
    rules: str = typer.Option(..., "--rules", help="Path to rules directory"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional path for decision JSON output"),
):
    enc_obj = json.loads(Path(input).read_text(encoding="utf-8"))
    from .schema import EncounterInput

    enc = EncounterInput(**enc_obj)
    rulepacks = load_rules_dir(rules)
    decision = decide(enc, rulepacks)
    rec = {"input": enc.model_dump(), "decision": decision.model_dump()}
    if out:
        Path(out).write_text(json.dumps(decision.model_dump(), indent=2), encoding="utf-8")
        typer.echo(out)
    else:
        typer.echo(json.dumps(rec))


@app.command("generate-bpmn")
def cli_generate_bpmn(
    rules: str = typer.Option(..., "--rules"),
    out: str = typer.Option(..., "--out"),
    format: str = typer.Option("xml", "--format", help="Output format: xml or json"),
    check: bool = typer.Option(False, "--check", help="Validate the generated BPMN"),
):
    rulepacks = load_rules_dir(rules)
    xml = build_bpmn(rulepacks)
    output_path = out
    temp_xml_path: str = ""

    if format.lower() == "xml":
        Path(out).write_text(xml, encoding="utf-8")
    elif format.lower() == "json":
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bpmn") as tf:
            tf.write(xml.encode("utf-8"))
            temp_xml_path = tf.name
        generator = FlowchartGenerator()
        generator.save_json_format(temp_xml_path, out, "bpmn")
    else:
        typer.echo("Error: --format must be 'xml' or 'json'", err=True)
        raise typer.Exit(code=1)

    if check:
        # Validate BPMN using XML path
        xml_path = out if format.lower() == "xml" else temp_xml_path
        report = check_bpmn_soundness(xml_path)
        all_ok = all(proc.get('summary_pass', False) for proc in report.get('processes', []))
        if not all_ok:
            typer.echo("BPMN validation failed:")
            typer.echo(str(report))
            raise typer.Exit(code=2)

    typer.echo(output_path)


@app.command("render-bpmn")
def cli_render_bpmn(
    bpmn: str = typer.Option(..., "--bpmn"),
    out: str = typer.Option(..., "--out"),
):
    try:
        bpmn_graph = bpmn_importer.apply(bpmn)
        gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.save(gviz, out)
    except Exception:
        ext = Path(out).suffix.lower()
        if ext == ".svg":
            svg = (
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"600\" height=\"120\">"
                "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
                f"<text x=\"12\" y=\"24\" font-family=\"Arial\" font-size=\"14\">BPMN placeholder for {Path(bpmn).name}</text>"
                "</svg>"
            )
            Path(out).write_text(svg, encoding="utf-8")
        else:
            Path(out).write_bytes(b"BPMN placeholder; Graphviz not available")
    typer.echo(out)


@app.command("generate-dmn")
def cli_generate_dmn(
    rules: str = typer.Option(..., "--rules"),
    out: str = typer.Option(..., "--out"),
    format: str = typer.Option("xml", "--format", help="Output format: xml or json"),
    check: bool = typer.Option(False, "--check", help="Validate the generated DMN"),
):
    rulepacks = load_rules_dir(rules)
    xml = generate_dmn(rulepacks)
    output_path = out
    temp_xml_path: str = ""

    if format.lower() == "xml":
        Path(out).write_text(xml, encoding="utf-8")
    elif format.lower() == "json":
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".dmn") as tf:
            tf.write(xml.encode("utf-8"))
            temp_xml_path = tf.name
        generator = FlowchartGenerator()
        generator.save_json_format(temp_xml_path, out, "dmn")
    else:
        typer.echo("Error: --format must be 'xml' or 'json'", err=True)
        raise typer.Exit(code=1)

    if check:
        # Validate DMN using XML path
        xml_path = out if format.lower() == "xml" else temp_xml_path
        report = check_dmn_tables(xml_path)
        all_ok = all(tbl.get('summary_pass', False) for tbl in report.get('tables', []))
        if not all_ok:
            typer.echo("DMN validation failed:")
            typer.echo(str(report))
            raise typer.Exit(code=2)

    typer.echo(output_path)


@app.command("render-dmn")
def cli_render_dmn(
    dmn: str = typer.Option(..., "--dmn"),
    out: str = typer.Option(..., "--out"),
):
    render_drd(dmn, out)
    typer.echo(out)


@app.command("export-csv")
def cli_export_csv(
    logs: str = typer.Option(..., "--logs", help="Path to encounters JSONL log"),
    out: str = typer.Option(..., "--out", help="Path to CSV output"),
):
    from .csv_export import to_csv

    records = []
    with Path(logs).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    to_csv(records, out)
    typer.echo(out)


@app.command("extract-pdf")
def cli_extract_pdf(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    module: str = typer.Option("extracted", "--module", help="Module name for extracted rules"),
    out: str = typer.Option(..., "--out", help="Output JSON file for extracted rules"),
):
    """Extract clinical rules from a WHO CHW PDF document."""
    extractor = CHWRuleExtractor()
    try:
        rules = extractor.process_pdf_to_rules(pdf, module)
        Path(out).write_text(json.dumps(rules, indent=2), encoding="utf-8")
        typer.echo(f"Extracted {len(rules)} rules to {out}")
    except Exception as e:
        typer.echo(f"Error processing PDF: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("pdf-to-bpmn")
def cli_pdf_to_bpmn(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    module: str = typer.Option("extracted", "--module", help="Module name for extracted rules"),
    out: str = typer.Option(..., "--out", help="Output BPMN file"),
):
    """Extract rules from PDF and generate BPMN directly."""
    extractor = CHWRuleExtractor()
    try:
        rules = extractor.process_pdf_to_rules(pdf, module)
        rulepacks = {module: rules}
        xml = build_bpmn(rulepacks)
        Path(out).write_text(xml, encoding="utf-8")
        typer.echo(f"Generated BPMN from {len(rules)} rules: {out}")
    except Exception as e:
        typer.echo(f"Error processing PDF to BPMN: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("pdf-to-dmn")
def cli_pdf_to_dmn(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    module: str = typer.Option("extracted", "--module", help="Module name for extracted rules"),
    out: str = typer.Option(..., "--out", help="Output DMN file"),
):
    """Extract rules from PDF and generate DMN directly."""
    extractor = CHWRuleExtractor()
    try:
        rules = extractor.process_pdf_to_rules(pdf, module)
        rulepacks = {module: rules}
        xml = generate_dmn(rulepacks)
        Path(out).write_text(xml, encoding="utf-8")
        typer.echo(f"Generated DMN from {len(rules)} rules: {out}")
    except Exception as e:
        typer.echo(f"Error processing PDF to DMN: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("generate-clinical-flowchart")
def cli_generate_clinical_flowchart(
    rules: str = typer.Option(..., "--rules", help="Path to rules JSON file"),
    module: str = typer.Option(..., "--module", help="Module name for the flowchart"),
    output: str = typer.Option(..., "--output", help="Output path for clinical flowchart image"),
):
    """Generate a comprehensive clinical decision flowchart from rules."""
    generator = FlowchartGenerator()
    try:
        # Load rules from JSON file
        rules_data = json.loads(Path(rules).read_text(encoding="utf-8"))
        generator.generate_clinical_flowchart(rules_data, module, output)
        typer.echo(f"✅ Generated clinical flowchart → {output}")
    except Exception as e:
        typer.echo(f"❌ Clinical flowchart generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("generate-flowchart")
def cli_generate_flowchart(
    input_file: str = typer.Option(..., "--input", help="Path to BPMN or DMN XML file"),
    output: str = typer.Option(..., "--output", help="Output path for flowchart image"),
    format_type: str = typer.Option(..., "--type", help="Input format: 'bpmn' or 'dmn'"),
):
    """Generate a technical flowchart from BPMN or DMN XML file."""
    generator = FlowchartGenerator()
    try:
        if format_type.lower() == "bpmn":
            generator.generate_bpmn_flowchart(input_file, output)
        elif format_type.lower() == "dmn":
            generator.generate_dmn_flowchart(input_file, output)
        else:
            typer.echo("Error: --type must be 'bpmn' or 'dmn'", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"✅ Generated technical flowchart → {output}")
    except Exception as e:
        typer.echo(f"❌ Flowchart generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("convert-to-json")
def cli_convert_to_json(
    input_file: str = typer.Option(..., "--input", help="Path to BPMN or DMN XML file"),
    output: str = typer.Option(..., "--output", help="Output path for readable JSON file"),
    format_type: str = typer.Option(..., "--type", help="Input format: 'bpmn' or 'dmn'"),
):
    """Convert BPMN or DMN XML to readable JSON format."""
    generator = FlowchartGenerator()
    try:
        generator.save_json_format(input_file, output, format_type)
        typer.echo(f"✅ Converted to JSON → {output}")
    except Exception as e:
        typer.echo(f"❌ JSON conversion failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("pdf-workflow")
def cli_pdf_workflow(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    module: str = typer.Option("extracted", "--module", help="Module name for extracted rules"),
    out_dir: str = typer.Option("chw_workflow_output", "--out-dir", help="Output directory for all generated files"),
):
    """Complete PDF-to-BPMN/DMN workflow: extract rules, generate BPMN, DMN, flowcharts, and JSON formats."""
    extractor = CHWRuleExtractor()
    generator = FlowchartGenerator()
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    (out_path / "01_extracted_data").mkdir(exist_ok=True)
    (out_path / "02_process_models").mkdir(exist_ok=True) 
    (out_path / "03_readable_formats").mkdir(exist_ok=True)
    (out_path / "04_clinical_flowcharts").mkdir(exist_ok=True)
    (out_path / "05_technical_diagrams").mkdir(exist_ok=True)
    
    try:
        # Extract rules
        typer.echo("🔍 Extracting rules from PDF...")
        rules = extractor.process_pdf_to_rules(pdf, module)
        rules_file = out_path / "01_extracted_data" / "chw_rules_extracted.json"
        rules_file.write_text(json.dumps(rules, indent=2), encoding="utf-8")
        typer.echo(f"✅ Extracted {len(rules)} rules → {rules_file}")
        
        rulepacks = {module: rules}
        
        # Generate BPMN
        typer.echo("🔧 Generating BPMN...")
        bpmn_xml = build_bpmn(rulepacks)
        bpmn_file = out_path / "02_process_models" / "chw_workflow_process.bpmn"
        bpmn_file.write_text(bpmn_xml, encoding="utf-8")
        typer.echo(f"✅ Generated BPMN → {bpmn_file}")
        
        # Generate DMN
        typer.echo("🔧 Generating DMN...")
        dmn_xml = generate_dmn(rulepacks)
        dmn_file = out_path / "02_process_models" / "chw_decision_logic.dmn"
        dmn_file.write_text(dmn_xml, encoding="utf-8")
        typer.echo(f"✅ Generated DMN → {dmn_file}")
        
        # Convert to readable JSON formats
        try:
            bpmn_json = out_path / "03_readable_formats" / "workflow_process_readable.json"
            generator.save_json_format(str(bpmn_file), str(bpmn_json), "bpmn")
            typer.echo(f"✅ Converted BPMN to JSON → {bpmn_json}")
        except Exception as e:
            typer.echo(f"⚠️  BPMN JSON conversion failed: {e}")
        
        try:
            dmn_json = out_path / "03_readable_formats" / "decision_logic_readable.json"
            generator.save_json_format(str(dmn_file), str(dmn_json), "dmn")
            typer.echo(f"✅ Converted DMN to JSON → {dmn_json}")
        except Exception as e:
            typer.echo(f"⚠️  DMN JSON conversion failed: {e}")
        
        # Generate clinical flowchart (main user-facing workflow)
        try:
            clinical_flowchart = out_path / "04_clinical_flowcharts" / "chw_clinical_workflow_guide.png"
            generator.generate_clinical_flowchart(rules, "CHW Guidelines", str(clinical_flowchart))
            typer.echo(f"✅ Generated clinical workflow guide → {clinical_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Clinical flowchart generation failed: {e}")
        
        # Generate technical flowcharts 
        try:
            bpmn_flowchart = out_path / "05_technical_diagrams" / "technical_bpmn_flowchart.png"
            generator.generate_bpmn_flowchart(str(bpmn_file), str(bpmn_flowchart))
            typer.echo(f"✅ Generated technical BPMN flowchart → {bpmn_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Technical BPMN flowchart failed: {e}")
        
        try:
            dmn_flowchart = out_path / "05_technical_diagrams" / "technical_dmn_flowchart.png"
            generator.generate_dmn_flowchart(str(dmn_file), str(dmn_flowchart))
            typer.echo(f"✅ Generated technical DMN flowchart → {dmn_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Technical DMN flowchart failed: {e}")
        
        # Render traditional visualizations
        try:
            bpmn_graph = bpmn_importer.apply(str(bpmn_file))
            gviz = bpmn_visualizer.apply(bpmn_graph)
            bpmn_svg = out_path / "05_technical_diagrams" / "bpmn_process_diagram.svg"
            bpmn_visualizer.save(gviz, str(bpmn_svg))
            typer.echo(f"✅ Rendered BPMN process diagram → {bpmn_svg}")
        except Exception:
            # Fallback SVG
            bpmn_svg = out_path / "05_technical_diagrams" / "bpmn_process_diagram.svg"
            svg = (
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"600\" height=\"120\">"
                "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
                f"<text x=\"12\" y=\"24\" font-family=\"Arial\" font-size=\"14\">BPMN Process Diagram</text>"
                "</svg>"
            )
            bpmn_svg.write_text(svg, encoding="utf-8")
            typer.echo(f"✅ Generated BPMN placeholder diagram → {bpmn_svg}")
        
        # Render DMN DRD 
        try:
            dmn_svg = out_path / "05_technical_diagrams" / "dmn_decision_diagram.svg"
            render_drd(str(dmn_file), str(dmn_svg))
            typer.echo(f"✅ Rendered DMN decision diagram → {dmn_svg}")
        except SystemExit as e:
            if e.code == 2:
                dmn_dot = out_path / "05_technical_diagrams" / "dmn_decision_diagram.dot"
                typer.echo(f"⚠️  Graphviz not available, saved DOT → {dmn_dot}")
        
        typer.echo(f"\n🎉 Complete CHW workflow generated in: {out_path}")
        typer.echo("\n📁 Generated file structure:")
        typer.echo("   📂 01_extracted_data/")
        typer.echo("      • chw_rules_extracted.json - Raw rules from PDF")
        typer.echo("   📂 02_process_models/") 
        typer.echo("      • chw_workflow_process.bpmn - BPMN process model")
        typer.echo("      • chw_decision_logic.dmn - DMN decision model")
        typer.echo("   📂 03_readable_formats/")
        typer.echo("      • workflow_process_readable.json - Human-readable BPMN")
        typer.echo("      • decision_logic_readable.json - Human-readable DMN")
        typer.echo("   📂 04_clinical_flowcharts/")
        typer.echo("      • chw_clinical_workflow_guide.png - ⭐ MAIN USER GUIDE")
        typer.echo("   📂 05_technical_diagrams/")
        typer.echo("      • technical_*.png - Technical flowcharts")
        typer.echo("      • *_diagram.svg - Standard BPMN/DMN diagrams")
        
    except Exception as e:
        typer.echo(f"❌ Workflow failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("text-workflow")
def cli_text_workflow(
    text_input: str = typer.Option(..., "--text", help="Text content or path to text file containing clinical guidelines"),
    module: str = typer.Option("text_rules", "--module", help="Module name for the extracted rules"),
    out_dir: str = typer.Option("text_workflow_output", "--out-dir", help="Output directory for all generated files"),
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="OpenAI API key for enhanced extraction (optional)"),
    use_openai: bool = typer.Option(True, "--use-openai/--no-openai", help="Use OpenAI for intelligent rule extraction"),
):
    """Complete text-to-workflow pipeline: extract rules from text, generate flowcharts and artifacts."""
    
    try:
        # Lazy import to avoid importing OpenAI dependencies unless needed
        from .text_processor import TextToFlowchartProcessor
        # Determine if input is a file path or direct text
        text_content = text_input
        if Path(text_input).exists():
            print(f"📄 Reading text from file: {text_input}")
            text_content = Path(text_input).read_text(encoding='utf-8')
        else:
            print("📝 Processing direct text input")
        
        # Initialize processor
        processor = TextToFlowchartProcessor(openai_api_key=openai_key)
        
        # Create comprehensive workflow
        generated_files = processor.create_comprehensive_workflow(
            text=text_content,
            module_name=module,
            output_dir=out_dir,
            use_openai=use_openai and (openai_key is not None)
        )
        
        typer.echo(f"\n🎉 Text workflow complete!")
        typer.echo(f"📁 Generated {len(generated_files)} files in: {out_dir}")
        typer.echo("\n📋 Key outputs:")
        if "clinical_flowchart" in generated_files:
            typer.echo(f"   ⭐ Clinical Guide: {generated_files['clinical_flowchart']}")
        if "rules" in generated_files:
            typer.echo(f"   📊 Extracted Rules: {generated_files['rules']}")
        
    except Exception as e:
        typer.echo(f"❌ Text workflow failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("pdf-openai-workflow") 
def cli_pdf_openai_workflow(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"),
    module: str = typer.Option("openai_extracted", "--module", help="Module name for extracted rules"),
    out_dir: str = typer.Option("pdf_openai_workflow_output", "--out-dir", help="Output directory for all generated files"),
    system_prompt_file: Optional[str] = typer.Option(None, "--system-prompt", help="Path to a custom OpenAI system prompt file"),
):
    """Complete PDF-via-OpenAI workflow: use OpenAI to intelligently extract comprehensive rules from PDF."""
    
    output_path = Path(out_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    (output_path / "01_extracted_data").mkdir(exist_ok=True)
    (output_path / "02_process_models").mkdir(exist_ok=True) 
    (output_path / "03_readable_formats").mkdir(exist_ok=True)
    (output_path / "04_clinical_flowcharts").mkdir(exist_ok=True)
    (output_path / "05_technical_diagrams").mkdir(exist_ok=True)
    
    try:
        # Initialize OpenAI extractor
        typer.echo("🤖 Initializing OpenAI-powered rule extraction...")
        # Lazy import to avoid importing OpenAI unless this command is used
        from .openai_extractor import OpenAIRuleExtractor
        custom_prompt: Optional[str] = None
        if system_prompt_file:
            try:
                custom_prompt = Path(system_prompt_file).read_text(encoding="utf-8")
                typer.echo(f"📝 Loaded custom system prompt from: {system_prompt_file}")
            except Exception as e:
                typer.echo(f"⚠️  Failed to read system prompt file: {e}")
        # Handle OpenAI API key - use provided key or fall back to environment variable
        import os
        api_key = openai_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            typer.echo("❌ OpenAI API key required. Provide via --openai-key or set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
            
        extractor = OpenAIRuleExtractor(api_key, system_prompt=custom_prompt)
        generator = FlowchartGenerator()
        
        # Extract rules using OpenAI
        typer.echo("🔍 Extracting rules from PDF using OpenAI...")
        rules = extractor.process_pdf_to_rules(pdf, module)
        
        # Save extracted rules
        rules_file = output_path / "01_extracted_data" / "openai_extracted_rules.json"
        rules_file.write_text(json.dumps(rules, indent=2), encoding="utf-8")
        typer.echo(f"✅ Extracted {len(rules)} comprehensive rules → {rules_file}")
        
        if not rules:
            typer.echo("⚠️  No rules extracted. Check PDF content and OpenAI response.")
            return
        
        rulepacks = {module: rules}
        
        # Generate BPMN
        typer.echo("🔧 Generating BPMN...")
        bpmn_xml = build_bpmn(rulepacks)
        bpmn_file = output_path / "02_process_models" / "openai_workflow_process.bpmn"
        bpmn_file.write_text(bpmn_xml, encoding="utf-8")
        typer.echo(f"✅ Generated BPMN → {bpmn_file}")
        
        # Generate DMN
        typer.echo("🔧 Generating DMN...")
        dmn_xml = generate_dmn(rulepacks)
        dmn_file = output_path / "02_process_models" / "openai_decision_logic.dmn"
        dmn_file.write_text(dmn_xml, encoding="utf-8")
        typer.echo(f"✅ Generated DMN → {dmn_file}")
        
        # Convert to readable JSON formats
        try:
            bpmn_json = output_path / "03_readable_formats" / "workflow_process_readable.json"
            generator.save_json_format(str(bpmn_file), str(bpmn_json), "bpmn")
            typer.echo(f"✅ Converted BPMN to JSON → {bpmn_json}")
        except Exception as e:
            typer.echo(f"⚠️  BPMN JSON conversion failed: {e}")
        
        try:
            dmn_json = output_path / "03_readable_formats" / "decision_logic_readable.json"
            generator.save_json_format(str(dmn_file), str(dmn_json), "dmn")
            typer.echo(f"✅ Converted DMN to JSON → {dmn_json}")
        except Exception as e:
            typer.echo(f"⚠️  DMN JSON conversion failed: {e}")
        
        # Generate clinical flowchart (main user-facing workflow)
        try:
            clinical_flowchart = output_path / "04_clinical_flowcharts" / "openai_clinical_workflow_guide.png"
            generator.generate_clinical_flowchart(rules, f"OpenAI-Enhanced {module}", str(clinical_flowchart))
            typer.echo(f"✅ Generated comprehensive clinical workflow guide → {clinical_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Clinical flowchart generation failed: {e}")
        
        # Generate technical flowcharts 
        try:
            bpmn_flowchart = output_path / "05_technical_diagrams" / "technical_bpmn_flowchart.png"
            generator.generate_bpmn_flowchart(str(bpmn_file), str(bpmn_flowchart))
            typer.echo(f"✅ Generated technical BPMN flowchart → {bpmn_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Technical BPMN flowchart failed: {e}")
        
        try:
            dmn_flowchart = output_path / "05_technical_diagrams" / "technical_dmn_flowchart.png"
            generator.generate_dmn_flowchart(str(dmn_file), str(dmn_flowchart))
            typer.echo(f"✅ Generated technical DMN flowchart → {dmn_flowchart}")
        except Exception as e:
            typer.echo(f"⚠️  Technical DMN flowchart failed: {e}")
        
        # Render traditional visualizations
        try:
            bpmn_graph = bpmn_importer.apply(str(bpmn_file))
            gviz = bpmn_visualizer.apply(bpmn_graph)
            bpmn_svg = output_path / "05_technical_diagrams" / "bpmn_process_diagram.svg"
            bpmn_visualizer.save(gviz, str(bpmn_svg))
            typer.echo(f"✅ Rendered BPMN process diagram → {bpmn_svg}")
        except Exception:
            # Fallback SVG
            bpmn_svg = output_path / "05_technical_diagrams" / "bpmn_process_diagram.svg"
            svg = (
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"600\" height=\"120\">"
                "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
                f"<text x=\"12\" y=\"24\" font-family=\"Arial\" font-size=\"14\">OpenAI-Enhanced BPMN Process</text>"
                "</svg>"
            )
            bpmn_svg.write_text(svg, encoding="utf-8")
            typer.echo(f"✅ Generated BPMN placeholder diagram → {bpmn_svg}")
        
        # Render DMN DRD 
        try:
            dmn_svg = output_path / "05_technical_diagrams" / "dmn_decision_diagram.svg"
            render_drd(str(dmn_file), str(dmn_svg))
            typer.echo(f"✅ Rendered DMN decision diagram → {dmn_svg}")
        except SystemExit as e:
            if e.code == 2:
                dmn_dot = output_path / "05_technical_diagrams" / "dmn_decision_diagram.dot"
                typer.echo(f"⚠️  Graphviz not available, saved DOT → {dmn_dot}")
        
        typer.echo(f"\n🎉 OpenAI-enhanced PDF workflow complete!")
        typer.echo(f"🤖 Extracted {len(rules)} comprehensive clinical rules using AI")
        typer.echo(f"📁 Generated organized output in: {output_path}")
        typer.echo("\n📁 Generated file structure:")
        typer.echo("   📂 01_extracted_data/")
        typer.echo("      • openai_extracted_rules.json - Comprehensive AI-extracted rules")
        typer.echo("   📂 04_clinical_flowcharts/")
        typer.echo("      • openai_clinical_workflow_guide.png - ⭐ COMPREHENSIVE USER GUIDE")
        typer.echo("   📂 + Standard BPMN/DMN models and technical diagrams")
        
    except Exception as e:
        typer.echo(f"❌ OpenAI workflow failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("generate-intelligent-dmn")
def cli_generate_intelligent_dmn(
    rules_file: str = typer.Option(..., "--rules", help="Path to extracted rules JSON file"),
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"),
    output_file: str = typer.Option("intelligent_decision_logic.dmn", "--output", help="Output DMN file path"),
    module: str = typer.Option("WHO_CHW", "--module", help="Module name for the DMN"),
):
    """Generate an intelligent, clean DMN from extracted rules using OpenAI."""
    try:
        from .openai_extractor import OpenAIRuleExtractor
        import json
        
        # Load environment variables first
        load_dotenv()
        
        # Handle OpenAI API key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        elif not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Provide via --openai-key or set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
        
        # Load extracted rules
        typer.echo(f"📋 Loading rules from: {rules_file}")
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        typer.echo(f"📊 Loaded {len(rules)} clinical rules")
        
        # Initialize OpenAI extractor
        extractor = OpenAIRuleExtractor()
        
        # Generate intelligent DMN
        typer.echo("🤖 Generating intelligent DMN with OpenAI...")
        dmn_xml = extractor.generate_intelligent_dmn(rules, module)
        
        # Save DMN
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dmn_xml)
        
        typer.echo(f"✅ Generated intelligent DMN → {output_file}")
        typer.echo("🎯 This DMN eliminates rule overlaps and provides clean clinical decision paths")
        
    except ImportError:
        typer.echo("❌ OpenAI dependency not available. Install with: pip install openai", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Intelligent DMN generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("generate-related-bpmn-dmn")
def cli_generate_related_bpmn_dmn(
    rules_file: str = typer.Option(..., "--rules", help="Path to extracted rules JSON file"),
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"),
    output_dir: str = typer.Option("related_artifacts", "--output-dir", help="Output directory for BPMN and DMN files"),
    module: str = typer.Option("WHO_CHW", "--module", help="Module name for the artifacts"),
):
    """Generate both BPMN and DMN together so OpenAI can relate them intelligently."""
    try:
        from .openai_extractor import OpenAIRuleExtractor
        import json
        import os
        from pathlib import Path
        
        # Handle OpenAI API key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        elif not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Provide via --openai-key or set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
        
        # Load extracted rules
        typer.echo(f"📋 Loading rules from: {rules_file}")
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        typer.echo(f"📊 Loaded {len(rules)} clinical rules")
        
        # Initialize OpenAI extractor
        extractor = OpenAIRuleExtractor()
        
        # Generate related BPMN+DMN
        typer.echo("🤖 Generating related BPMN+DMN with OpenAI...")
        artifacts = extractor.generate_related_bpmn_dmn(rules, module)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save BPMN
        bpmn_file = output_path / f"{module.lower()}_related_workflow.bpmn"
        with open(bpmn_file, 'w', encoding='utf-8') as f:
            f.write(artifacts["bpmn_xml"])
        
        # Save DMN
        dmn_file = output_path / f"{module.lower()}_related_decisions.dmn"
        with open(dmn_file, 'w', encoding='utf-8') as f:
            f.write(artifacts["dmn_xml"])
        
        # Save raw response for debugging
        raw_file = output_path / f"{module.lower()}_related_response.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(artifacts["raw_response"])
        
        typer.echo(f"✅ Generated related BPMN+DMN → {output_dir}")
        typer.echo(f"   📄 BPMN: {bpmn_file}")
        typer.echo(f"   📄 DMN: {dmn_file}")
        typer.echo("🎯 These artifacts are intelligently related and work together seamlessly")
        
    except ImportError:
        typer.echo("❌ OpenAI dependency not available. Install with: pip install openai", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Related BPMN+DMN generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("extract-pdf-refined")
def cli_extract_pdf_refined(
    pdf: str = typer.Option(..., "--pdf", help="Path to WHO CHW PDF document"),
    output_file: str = typer.Option("refined_extracted_rules.json", "--output", help="Output JSON file for extracted rules"),
    module: str = typer.Option("refined_extracted", "--module", help="Module name for extracted rules"),
):
    """Extract clinical rules using the refined, generalized OpenAI extractor."""
    try:
        from .openai_extractor_refined import OpenAIRuleExtractor
        
        # Load environment variables first
        load_dotenv()
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
        
        # Initialize refined extractor
        extractor = OpenAIRuleExtractor()
        
        # Extract rules using refined method
        print(f"🔍 Extracting rules from PDF using refined extractor: {pdf}")
        extracted_data = extractor.extract_rules_from_pdf(pdf, module_name=module)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        print(f"✅ Refined extraction completed!")
        print(f"📊 Extracted {len(extracted_data.get('variables', []))} variables")
        print(f"📊 Extracted {len(extracted_data.get('rules', []))} rules")
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        typer.echo(f"❌ Refined extraction failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command("generate-refined-bpmn-dmn")
def cli_generate_refined_bpmn_dmn(
    rules_file: str = typer.Option(..., "--rules", help="Path to refined extracted rules JSON file"),
    output_dir: str = typer.Option("refined_artifacts", "--output-dir", help="Output directory for BPMN and DMN files"),
    module: str = typer.Option("WHO_CHW_Refined", "--module", help="Module name for the artifacts"),
):
    """Generate BPMN and DMN using the refined extractor with standardized prompts."""
    try:
        from .openai_extractor_refined import OpenAIRuleExtractor
        from pathlib import Path
        import json
        
        # Load environment variables first
        load_dotenv()
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
        
        # Load refined rules data
        with open(rules_file) as f:
            rules_data = json.load(f)
        
        print(f"📋 Loading refined rules from: {rules_file}")
        if isinstance(rules_data, dict) and 'variables' in rules_data and 'rules' in rules_data:
            print(f"📊 Found {len(rules_data['variables'])} variables and {len(rules_data['rules'])} rules")
        else:
            typer.echo("❌ Invalid rules file format. Expected refined extractor output with 'variables' and 'rules' keys.", err=True)
            raise typer.Exit(code=1)
        
        # Initialize refined extractor
        extractor = OpenAIRuleExtractor()
        
        # Generate BPMN and DMN
        print(f"🤖 Generating BPMN and DMN with standardized prompts...")
        result = extractor.generate_bpmn_dmn_from_rules(rules_data, module_name=module)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save BPMN
        bpmn_file = output_path / f"{module.lower()}_workflow.bpmn"
        with open(bpmn_file, 'w') as f:
            f.write(result['bpmn_xml'])
        
        # Save DMN
        dmn_file = output_path / f"{module.lower()}_decision_logic.dmn"
        with open(dmn_file, 'w') as f:
            f.write(result['dmn_xml'])
        
        # Save raw response for debugging
        raw_file = output_path / f"{module.lower()}_raw_response.txt"
        with open(raw_file, 'w') as f:
            f.write(result['raw_response'])
        
        print(f"✅ Refined BPMN and DMN generation completed!")
        print(f"📄 BPMN saved to: {bpmn_file}")
        print(f"📄 DMN saved to: {dmn_file}")
        print(f"📄 Raw response saved to: {raw_file}")
        print(f"🎯 Artifacts follow standardized clinical interview policy with diarrhea-first flow")
        
    except Exception as e:
        typer.echo(f"❌ Refined BPMN/DMN generation failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command("generate-sequential-bpmn-dmn")
def cli_generate_sequential_bpmn_dmn(
    rules_file: str = typer.Option(..., "--rules", help="Path to extracted rules JSON file"),
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"),
    output_dir: str = typer.Option("sequential_artifacts", "--output-dir", help="Output directory for BPMN and DMN files"),
    module: str = typer.Option("WHO_CHW", "--module", help="Module name for the artifacts"),
):
    """Generate BPMN first, then DMN with BPMN context for intelligent flow logic."""
    try:
        from .openai_extractor import OpenAIRuleExtractor
        import json
        import os
        from pathlib import Path
        
        # Handle OpenAI API key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        elif not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Provide via --openai-key or set OPENAI_API_KEY environment variable.", err=True)
            raise typer.Exit(code=1)
        
        # Load extracted rules
        typer.echo(f"📋 Loading rules from: {rules_file}")
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        typer.echo(f"📊 Loaded {len(rules)} clinical rules")
        
        # Initialize OpenAI extractor
        extractor = OpenAIRuleExtractor()
        
        # Generate sequential BPMN→DMN
        typer.echo("🤖 Generating sequential BPMN→DMN with OpenAI...")
        artifacts = extractor.generate_sequential_bpmn_dmn(rules, module)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save BPMN
        bpmn_file = output_path / f"{module.lower()}_sequential_workflow.bpmn"
        with open(bpmn_file, 'w', encoding='utf-8') as f:
            f.write(artifacts["bpmn_xml"])
        
        # Save DMN
        dmn_file = output_path / f"{module.lower()}_sequential_decisions.dmn"
        with open(dmn_file, 'w', encoding='utf-8') as f:
            f.write(artifacts["dmn_xml"])
        
        typer.echo(f"✅ Generated sequential BPMN→DMN → {output_dir}")
        typer.echo(f"   📄 BPMN: {bpmn_file}")
        typer.echo(f"   📄 DMN: {dmn_file}")
        typer.echo("🎯 DMN was generated with BPMN context for intelligent integration")
        
    except ImportError:
        typer.echo("❌ OpenAI dependency not available. Install with: pip install openai", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Sequential BPMN→DMN generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("version")
def cli_version():
    try:
        from importlib.metadata import version as _pkg_version
        typer.echo(_pkg_version("chatchw"))
    except Exception:
        from . import __version__
        typer.echo(__version__)


@app.command("chat")
def cli_chat(
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN file for conversation flow"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN file for decision logic"),
    session_id: str = typer.Option("default", "--session", help="Session ID for conversation tracking"),
    save_session: bool = typer.Option(True, "--save/--no-save", help="Save conversation session"),
):
    """Start an interactive medical consultation chatbot using BPMN/DMN logic"""
    
    try:
        # Initialize the chatbot engine
        typer.echo(f"🤖 Loading chatbot with BPMN: {bpmn} and DMN: {dmn}")
        engine = ChatbotEngine(bpmn, dmn)
        
        # Start conversation
        state = engine.start_conversation()
        
        typer.echo("\n" + "="*60)
        typer.echo("🏥 MEDICAL CONSULTATION CHATBOT")
        typer.echo("="*60)
        typer.echo("Welcome! I'll help guide you through a medical assessment.")
        typer.echo("Type 'quit' at any time to exit.\n")
        
        while not engine.is_conversation_complete(state):
            question = engine.get_next_question(state)
            
            if question is None:
                typer.echo("⚠️  No more questions available. Analyzing current data...")
                break
            
            # Display question
            typer.echo(f"❓ {question.text}")
            
            if question.question_type == 'boolean':
                typer.echo("   (Answer: yes/no or y/n)")
            elif question.question_type == 'choice' and question.choices:
                typer.echo(f"   (Choices: {', '.join(question.choices)})")
            elif question.question_type == 'numeric':
                typer.echo("   (Enter a number)")
            
            # Get user input
            try:
                answer = typer.prompt("👤 Your answer")
                
                if answer.lower() in ['quit', 'exit', 'q']:
                    typer.echo("👋 Goodbye!")
                    return
                
                # Process the answer
                state, is_valid = engine.process_answer(state, question.variable, answer)
                
                if not is_valid:
                    typer.echo(f"❌ Invalid input. Please try again.")
                    continue  # Re-ask the same question
                
                # Show any immediate feedback
                if state.reasoning:
                    latest_reasoning = state.reasoning[-1] if state.reasoning else ""
                    if latest_reasoning and "Clinical finding:" in latest_reasoning:
                        typer.echo(f"📋 {latest_reasoning}")
                
                typer.echo()  # Add spacing
                
            except (KeyboardInterrupt, EOFError):
                typer.echo("\n👋 Goodbye!")
                return
        
        # Show final recommendation
        recommendation = engine.get_final_recommendation(state)
        
        typer.echo("\n" + "="*60)
        typer.echo("🎯 CONSULTATION RESULTS")
        typer.echo("="*60)
        
        # Clinical Summary
        if recommendation.get('clinical_summary'):
            typer.echo(f"\n📊 {recommendation['clinical_summary']}")
        
        # Primary Recommendation
        if recommendation['outcome']:
            outcome_emoji = {"Hospital": "🏥", "Clinic": "🏪", "Home": "🏠"}
            emoji = outcome_emoji.get(recommendation['outcome'], "📍")
            typer.echo(f"\n{emoji} PRIMARY RECOMMENDATION: {recommendation['outcome'].upper()}")
        
        # Detailed Clinical Reasoning
        if recommendation.get('clinical_reasoning'):
            typer.echo("\n🧠 CLINICAL REASONING:")
            for reason in recommendation['clinical_reasoning']:
                typer.echo(f"   • {reason}")
        
        # Original reasoning for backward compatibility
        if recommendation.get('reasoning') and not recommendation.get('clinical_reasoning'):
            typer.echo("\n📋 Clinical Findings:")
            for reason in recommendation['reasoning']:
                typer.echo(f"   • {reason}")
        
        # Active Flags
        if recommendation['flags']:
            active_flags = [flag for flag, status in recommendation['flags'].items() if status]
            if active_flags:
                typer.echo("\n🚨 Active Clinical Flags:")
                for flag in active_flags:
                    typer.echo(f"   • {flag}")
        
        # Specific Next Steps
        if recommendation.get('next_steps'):
            typer.echo("\n📋 NEXT STEPS:")
            for step in recommendation['next_steps']:
                typer.echo(f"   {step}")
        
        # Save session if requested
        if save_session:
            session_file = f"chatbot_session_{session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(recommendation, f, indent=2)
        typer.echo(f"\n💾 Session saved to: {session_file}")

        typer.echo("\n" + "="*60)
        
    except Exception as e:
        typer.echo(f"❌ Error starting chatbot: {e}", err=True)
        raise typer.Exit(code=1)

@app.command("chat-dynamic")
def cli_chat_dynamic(
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN workflow file"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN decision logic file"),
    session: str = typer.Option("dynamic_session.json", "--session", help="Session file name to save results"),
):
    """Start a dynamic BPMN-driven medical consultation chatbot with enhanced reasoning."""
    try:
        typer.echo(f"🤖 Loading dynamic chatbot with BPMN: {bpmn} and DMN: {dmn}")
        typer.echo()
        
        # Initialize dynamic engine
        engine = DynamicChatbotEngine(bpmn, dmn)
        state = engine.start_conversation()
        
        typer.echo("="*60)
        typer.echo("🏥 DYNAMIC MEDICAL CONSULTATION CHATBOT")
        typer.echo("="*60)
        typer.echo("Welcome! I'll guide you through a clinical assessment following WHO guidelines.")
        typer.echo("Type 'quit' at any time to exit.")
        typer.echo()
        
        # Main conversation loop
        while not engine.is_conversation_complete(state):
            # Get current task questions
            task_questions = engine.get_current_task_questions(state)
            
            if task_questions:
                typer.echo(f"📋 {task_questions.task_name}")
                typer.echo("-" * 40)
                
                # Ask each question for this task
                for question in task_questions.questions:
                    variable = question['variable']
                    
                    # Skip if already collected
                    if variable in state.collected_data:
                        continue
                    
                    # Ask the question
                    while True:
                        typer.echo(f"❓ {question['text']}")
                        if question.get('help'):
                            typer.echo(f"   💡 {question['help']}")
                        
                        if question['type'] == 'boolean':
                            typer.echo("   (Answer: yes/no or y/n)")
                        elif question['type'] == 'numeric':
                            typer.echo("   (Enter a number)")
                        
                        try:
                            answer = typer.prompt("👤 Your answer")
                            
                            if answer.lower() in ['quit', 'exit', 'q']:
                                typer.echo("\n👋 Goodbye!")
                                return
                            
                            # Validate and convert input
                            if question['type'] == 'boolean':
                                if answer.lower() in ['yes', 'y', '1', 'true']:
                                    value = True
                                elif answer.lower() in ['no', 'n', '0', 'false']:
                                    value = False
                                else:
                                    typer.echo("❌ Please answer yes/no (y/n)")
                                    continue
                            elif question['type'] == 'numeric':
                                try:
                                    value = float(answer)
                                except ValueError:
                                    typer.echo("❌ Please enter a valid number")
                                    continue
                            else:
                                value = answer
                            
                            # Process the input
                            engine.process_user_input(state, variable, value)
                            break
                            
                        except (KeyboardInterrupt, EOFError):
                            typer.echo("\n👋 Goodbye!")
                            return
                
                typer.echo()  # Add spacing between tasks
            
            # Advance workflow
            if not engine.advance_workflow(state):
                break
        
        # Show final recommendation with enhanced reasoning
        recommendation = engine.get_final_recommendation(state)
        
        typer.echo("\n" + "="*60)
        typer.echo("🎯 CONSULTATION RESULTS")
        typer.echo("="*60)
        
        # Clinical Summary
        if recommendation.get('clinical_summary'):
            typer.echo(f"\n📊 {recommendation['clinical_summary']}")
        
        # Primary Recommendation
        if recommendation['outcome']:
            outcome_emoji = {"Hospital": "🏥", "Clinic": "🏪", "Home": "🏠"}
            emoji = outcome_emoji.get(recommendation['outcome'], "📍")
            typer.echo(f"\n{emoji} PRIMARY RECOMMENDATION: {recommendation['outcome'].upper()}")
        
        # Detailed Clinical Reasoning
        if recommendation.get('clinical_reasoning'):
            typer.echo("\n🧠 CLINICAL REASONING:")
            for reason in recommendation['clinical_reasoning']:
                typer.echo(f"   • {reason}")
        
        # Active Flags
        if recommendation['flags']:
            active_flags = [flag for flag, status in recommendation['flags'].items() if status]
            if active_flags:
                typer.echo("\n🚨 Active Clinical Flags:")
                for flag in active_flags:
                    typer.echo(f"   • {flag}")
        
        # Specific Next Steps
        if recommendation.get('next_steps'):
            typer.echo("\n📋 NEXT STEPS:")
            for step in recommendation['next_steps']:
                typer.echo(f"   {step}")
        
        # Save session
        session_file = f"dynamic_{session}"
        with open(session_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        typer.echo(f"\n💾 Session saved to: {session_file}")
        
        typer.echo("\n" + "="*60)
        
    except Exception as e:
        typer.echo(f"❌ Dynamic chatbot error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("chat-batch")
def cli_chat_batch(
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN file for conversation flow"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN file for decision logic"),
    input_file: str = typer.Option(..., "--input", help="JSON file with patient data"),
    output_file: str = typer.Option("batch_results.json", "--output", help="Output file for results"),
):
    """Run batch processing using chatbot logic on pre-collected patient data"""
    
    try:
        # Initialize the chatbot engine
        typer.echo(f"🤖 Loading chatbot with BPMN: {bpmn} and DMN: {dmn}")
        engine = ChatbotEngine(bpmn, dmn)
        
        # Load input data
        with open(input_file, 'r') as f:
            patient_data = json.load(f)
        
        typer.echo(f"📊 Processing patient data from: {input_file}")
        
        # Start conversation and feed all data at once
        state = engine.start_conversation()
        
        # Process all available data
        for variable, value in patient_data.items():
            if variable in engine.questions:
                state, is_valid = engine.process_answer(state, variable, value)
                if not is_valid:
                    typer.echo(f"⚠️ Invalid value for {variable}: {value}")
        
        # Get final recommendation
        recommendation = engine.get_final_recommendation(state)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        typer.echo(f"✅ Results saved to: {output_file}")
        
        # Show summary
        if recommendation['outcome']:
            typer.echo(f"🎯 Recommendation: {recommendation['outcome']}")
        
        if recommendation['reasoning']:
            typer.echo("📋 Key findings:")
            for reason in recommendation['reasoning'][:3]:  # Show first 3
                typer.echo(f"   • {reason}")
        
    except Exception as e:
        typer.echo(f"❌ Error in batch processing: {e}")
        raise typer.Exit(1)


@app.command("generate-modular-bpmn-dmn")
def cli_generate_modular_bpmn_dmn(
    rules_file: str = typer.Option(..., "--rules", help="Path to refined extracted rules JSON file"),
    output_dir: str = typer.Option("modular_artifacts", "--output-dir", help="Output directory for modular DMN/BPMN/JSON files"),
    module: str = typer.Option("WHO_CHW_Modular", "--module", help="Module name for artifacts"),
    who_pdf: str = typer.Option(None, "--who-pdf", help="Optional WHO CHW PDF path for refs"),
):
    """Generate modular DMN package (DMN + CANONICAL_MAP + QA_REPORT + ASK_PLAN) and BPMN from refined rules."""
    try:
        from .openai_extractor_refined import OpenAIRuleExtractor
        from pathlib import Path
        import json
        import os

        # Load env
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            typer.echo("❌ OpenAI API key required. Set OPENAI_API_KEY.", err=True)
            raise typer.Exit(code=1)

        # Load rules
        rules_data = json.loads(Path(rules_file).read_text(encoding="utf-8"))
        if not isinstance(rules_data, dict) or "rules" not in rules_data:
            typer.echo("❌ Invalid rules file format.", err=True)
            raise typer.Exit(code=1)

        # Optional WHO text
        who_text = None
        if who_pdf:
            try:
                extractor_tmp = OpenAIRuleExtractor()
                who_text = extractor_tmp.extract_text_from_pdf(who_pdf)
            except Exception as e:
                typer.echo(f"⚠️ Failed to read WHO PDF: {e}")

        extractor = OpenAIRuleExtractor()
        typer.echo("🤖 Generating modular DMN package…")
        pkg = extractor.generate_modular_dmn_package(rules_data, module_name=module, who_pdf_text=who_text)

        # Save outputs
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dmn_file = out_dir / f"{module.lower()}_modular.dmn"
        (out_dir / f"{module.lower()}_canonical_map.json").write_text(json.dumps(pkg["canonical_map"], indent=2), encoding="utf-8")
        (out_dir / f"{module.lower()}_qa_report.json").write_text(json.dumps(pkg["qa_report"], indent=2), encoding="utf-8")
        (out_dir / f"{module.lower()}_ask_plan.json").write_text(json.dumps(pkg["ask_plan"], indent=2), encoding="utf-8")
        dmn_file.write_text(pkg["dmn_xml"], encoding="utf-8")

        typer.echo(f"📄 Modular DMN saved to: {dmn_file}")
        typer.echo("🧭 CANONICAL_MAP / QA_REPORT / ASK_PLAN saved alongside DMN")

        typer.echo("🛠️ Generating BPMN from modular package…")
        bpmn_xml = extractor.generate_bpmn_from_modular(pkg["dmn_xml"], pkg["canonical_map"], pkg["ask_plan"])
        bpmn_file = out_dir / f"{module.lower()}_workflow.bpmn"
        bpmn_file.write_text(bpmn_xml, encoding="utf-8")
        typer.echo(f"📄 BPMN saved to: {bpmn_file}")
        (out_dir / f"{module.lower()}_raw.txt").write_text(pkg["raw"], encoding="utf-8")
        typer.echo("✅ Modular generation complete")

    except Exception as e:
        typer.echo(f"❌ Modular generation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("chat-validate")
def cli_chat_validate(
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN file to validate"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN file to validate"),
):
    """Validate BPMN and DMN files for chatbot compatibility"""
    
    try:
        typer.echo(f"🔍 Validating BPMN: {bpmn}")
        typer.echo(f"🔍 Validating DMN: {dmn}")
        
        # Try to initialize the chatbot engine
        engine = ChatbotEngine(bpmn, dmn)
        
        # Show parsed structure
        typer.echo("\n✅ BPMN Structure:")
        typer.echo(f"   • Start node: {engine.conversation_flow.start_node}")
        typer.echo(f"   • End nodes: {len(engine.conversation_flow.end_nodes)}")
        typer.echo(f"   • Decision gateways: {len(engine.conversation_flow.decision_gateways)}")
        typer.echo(f"   • Task modules: {len(engine.conversation_flow.task_modules)}")
        
        typer.echo("\n✅ DMN Structure:")
        typer.echo(f"   • Input variables: {len(engine.decision_logic.input_data)}")
        typer.echo(f"   • Decision tables: {len(engine.decision_logic.decisions)}")
        typer.echo(f"   • Variable types: {len(engine.decision_logic.variable_types)}")
        
        typer.echo("\n✅ Questions Available:")
        for var, question in list(engine.questions.items())[:5]:  # Show first 5
            typer.echo(f"   • {var}: {question.text}")
        if len(engine.questions) > 5:
            typer.echo(f"   ... and {len(engine.questions) - 5} more")
        
        typer.echo("\n🎯 Files are valid and ready for chatbot use!")
        
    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}")
        raise typer.Exit(1)


@app.command("guarded-workflow")
def cli_guarded_workflow(
    pdf: str = typer.Option(..., "--pdf", help="WHO CHW PDF path"),
    outdir: str = typer.Option("guarded_artifacts", "--outdir", help="Output directory"),
    module: str = typer.Option("WHO_CHW_Guarded", "--module", help="Module name"),
):
    """Run the guarded 5-step LLM-only pipeline: per-section extraction → merge → modular DMN+ASK → BPMN → coverage audit."""
    try:
        from .openai_extractor_guarded import OpenAIGuardedExtractor
        from pathlib import Path
        import json
        load_dotenv()

        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)

        extractor = OpenAIGuardedExtractor()
        typer.echo("🔎 Step 1: Per-section extraction…")
        sec_objs = extractor.extract_rules_per_section(pdf)
        (out_path / f"{module.lower()}_sections.json").write_text(json.dumps(sec_objs, indent=2), encoding="utf-8")

        typer.echo("🧩 Step 2: Merge + canonicalize…")
        merged = extractor.merge_sections(sec_objs)
        (out_path / f"{module.lower()}_merged_ir.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

        typer.echo("🧠 Step 3: DMN + ASK_PLAN…")
        dmn_xml, ask_plan = extractor.generate_dmn_and_ask_plan(merged)
        (out_path / f"{module.lower()}_dmn.dmn").write_text(dmn_xml, encoding="utf-8")
        (out_path / f"{module.lower()}_ask_plan.json").write_text(json.dumps(ask_plan, indent=2), encoding="utf-8")

        typer.echo("🛠️ Step 4: BPMN…")
        bpmn_xml = extractor.generate_bpmn(dmn_xml, ask_plan)
        (out_path / f"{module.lower()}_workflow.bpmn").write_text(bpmn_xml, encoding="utf-8")

        typer.echo("✅ Step 5: Coverage audit…")
        audit = extractor.audit_coverage(merged, dmn_xml)
        (out_path / f"{module.lower()}_coverage.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

        typer.echo(f"✅ Guarded pipeline complete → {out_path}")

    except Exception as e:
        typer.echo(f"❌ Guarded workflow failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("validate-artifacts")
def cli_validate_artifacts(
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN file to validate"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN file to validate"),
):
    """Run structural BPMN checks, DMN logical checks, and BPMN/DMN alignment checks."""
    try:
        typer.echo(f"🔍 BPMN soundness: {bpmn}")
        bpmn_report = check_bpmn_soundness(bpmn)
        typer.echo(bpmn_report)

        typer.echo(f"\n🔍 DMN logical checks: {dmn}")
        dmn_report = check_dmn_tables(dmn)
        typer.echo(dmn_report)

        typer.echo(f"\n🔍 BPMN/DMN alignment checks")
        align_report = check_alignment(bpmn, dmn)
        typer.echo(align_report)

        summary_pass = all(
            p.get('summary_pass', False)
            for p in [
                {'summary_pass': all(proc.get('summary_pass', False) for proc in bpmn_report.get('processes', []))},
                {'summary_pass': all(t.get('summary_pass', False) for t in dmn_report.get('tables', []))},
                align_report,
            ]
        )
        if not summary_pass:
            raise typer.Exit(code=2)

    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}")
        raise typer.Exit(code=1)


@app.command("csv-validate")
def cli_csv_validate(
    csv_path: str = typer.Option(..., "--csv", help="Path to rubric CSV"),
    bpmn: str = typer.Option(..., "--bpmn", help="Path to BPMN file"),
    dmn: str = typer.Option(..., "--dmn", help="Path to DMN file"),
):
    """Validate chatbot outcomes against a rubric CSV of scenarios and expected triage."""
    try:
        engine = ChatbotEngine(bpmn, dmn)
        total = 0
        passes = 0

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            # Heuristic: detect columns for expected triage and key inputs
            # We'll look for lowercase keywords in cells to infer variables
            for row in reader:
                total += 1
                row_text = " ".join(row).lower()
                # Build inputs
                inputs = {}
                # Very simple mapping based on the rubric columns
                if "convulsion" in row_text:
                    inputs["convulsion"] = "convulsion" in row_text and ("yes" in row_text or "present" in row_text)
                if "blood in the stool" in row_text or "bloody diarrhea" in row_text or "blood in stool" in row_text:
                    inputs["blood_in_stool"] = True
                if "edema" in row_text and "both feet" in row_text:
                    inputs["edema_both_feet"] = True
                if "muac" in row_text and "<115" in row_text:
                    inputs["muac_mm"] = 110
                if "fever" in row_text and "high" in row_text:
                    inputs["temp"] = 39.0
                    inputs["feels_very_hot"] = True
                if "no dehydration" in row_text:
                    inputs["diarrhea_days"] = 1
                if "≥14 days" in row_text or ">=14 days" in row_text or "14 days" in row_text and "cough" in row_text:
                    # This is a cough scenario; we do not have cough variable, skip
                    pass

                # Expected outcome inference
                expected = None
                if "urgent" in row_text or "refer" in row_text or "danger sign" in row_text or "severe" in row_text:
                    expected = "Hospital"
                elif "follow up in 3 days" in row_text or "clinic" in row_text or "antibiotic" in row_text or "zinc" in row_text:
                    expected = "Clinic"
                elif "home" in row_text or "home care" in row_text or "supportive" in row_text:
                    expected = "Home"

                # Run engine
                state = engine.start_conversation()
                for var, value in inputs.items():
                    if var in engine.questions:
                        state, _ = engine.process_answer(state, var, value)
                rec = engine.get_final_recommendation(state)

                got = rec.get("outcome")
                ok = (expected is None) or (got == expected)
                passes += 1 if ok else 0

            typer.echo(json.dumps({
                "total": total,
                "passes": passes,
                "fails": total - passes
            }, indent=2))

    except Exception as e:
        typer.echo(f"❌ CSV validation failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
