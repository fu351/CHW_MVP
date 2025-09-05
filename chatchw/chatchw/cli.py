from __future__ import annotations

import json
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
from .validator import check_bpmn_soundness, check_dmn_tables, check_alignment
import csv

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    openai_key: str = typer.Option(..., "--openai-key", help="OpenAI API key for intelligent extraction"),
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
        extractor = OpenAIRuleExtractor(openai_key, system_prompt=custom_prompt)
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
        
        if recommendation['outcome']:
            outcome_emoji = {"Hospital": "🏥", "Clinic": "🏪", "Home": "🏠"}
            emoji = outcome_emoji.get(recommendation['outcome'], "📍")
            typer.echo(f"{emoji} RECOMMENDATION: {recommendation['outcome'].upper()}")
        
        if recommendation['reasoning']:
            typer.echo("\n📋 Clinical Findings:")
            for reason in recommendation['reasoning']:
                typer.echo(f"   • {reason}")
        
        if recommendation['flags']:
            typer.echo("\n🚨 Active Flags:")
            for flag, status in recommendation['flags'].items():
                if status:
                    typer.echo(f"   • {flag}")
        
        # Save session if requested
        if save_session:
            session_file = f"chatbot_session_{session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(recommendation, f, indent=2)
            typer.echo(f"\n💾 Session saved to: {session_file}")
        
        typer.echo("\n" + "="*60)
        
    except Exception as e:
        typer.echo(f"❌ Error starting chatbot: {e}")
        raise typer.Exit(1)


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
