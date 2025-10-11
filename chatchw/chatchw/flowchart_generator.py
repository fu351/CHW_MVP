from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
except ImportError:
    plt = None
    patches = None
    nx = None


class FlowchartGenerator:
    """Generate flowcharts from BPMN and DMN artifacts."""
    
    def __init__(self):
        self.bpmn_ns = "http://www.omg.org/spec/BPMN/20100524/MODEL"
        self.dmn_ns = "https://www.omg.org/spec/DMN/20191111/MODEL/"
    
    def parse_bpmn_to_json(self, bpmn_path: str) -> Dict:
        """Parse BPMN XML and convert to readable JSON format."""
        tree = ET.parse(bpmn_path)
        root = tree.getroot()
        
        # Find process elements
        processes = []
        for process in root.findall(f".//{{{self.bpmn_ns}}}process"):
            proc_data = {
                "id": process.get("id", "unknown"),
                "name": process.get("name", ""),
                "isExecutable": process.get("isExecutable", "false"),
                "elements": []
            }
            
            # Find all elements in the process
            for elem in process:
                elem_type = elem.tag.split("}")[-1]  # Remove namespace
                elem_data = {
                    "type": elem_type,
                    "id": elem.get("id", ""),
                    "name": elem.get("name", "")
                }
                
                # Add specific attributes based on element type
                if elem_type == "sequenceFlow":
                    elem_data["sourceRef"] = elem.get("sourceRef", "")
                    elem_data["targetRef"] = elem.get("targetRef", "")
                
                proc_data["elements"].append(elem_data)
            
            processes.append(proc_data)
        
        return {
            "type": "BPMN",
            "processes": processes
        }
    
    def parse_dmn_to_json(self, dmn_path: str) -> Dict:
        """Parse DMN XML and convert to readable JSON format."""
        tree = ET.parse(dmn_path)
        root = tree.getroot()
        
        # Find input data
        input_data = []
        for inp in root.findall(f".//{{{self.dmn_ns}}}inputData"):
            var = inp.find(f".//{{{self.dmn_ns}}}variable")
            input_data.append({
                "id": inp.get("id", ""),
                "name": inp.get("name", ""),
                "variable": (var.get("name") if var is not None else inp.get("name", "")),
                "typeRef": (var.get("typeRef") if var is not None else "")
            })
        
        # Find decisions
        decisions = []
        for decision in root.findall(f".//{{{self.dmn_ns}}}decision"):
            dec_data = {
                "id": decision.get("id", ""),
                "name": decision.get("name", ""),
                "requirements": []
            }
            
            # Find information requirements
            for req in decision.findall(f".//{{{self.dmn_ns}}}informationRequirement"):
                req_input = req.find(f".//{{{self.dmn_ns}}}requiredInput")
                if req_input is not None:
                    href = req_input.get("href", "")
                    if href.startswith("#"):
                        dec_data["requirements"].append(href[1:])
            
            # Find decision table
            table = decision.find(f".//{{{self.dmn_ns}}}decisionTable")
            if table is not None:
                dec_data["hitPolicy"] = table.get("hitPolicy", "")
                dec_data["inputs"] = []
                dec_data["outputs"] = []
                dec_data["rules"] = []
                
                # Parse inputs
                for inp in table.findall(f".//{{{self.dmn_ns}}}input"):
                    inp_expr = inp.find(f".//{{{self.dmn_ns}}}inputExpression")
                    if inp_expr is not None:
                        # Prefer nested dmn:text if present
                        text_el = inp_expr.find(f".//{{{self.dmn_ns}}}text")
                        expr_text = (text_el.text if text_el is not None else (inp_expr.text or ""))
                        dec_data["inputs"].append(expr_text)
                
                # Parse outputs
                for out in table.findall(f".//{{{self.dmn_ns}}}output"):
                    dec_data["outputs"].append(out.get("name", ""))
                
                # Parse rules
                for rule in table.findall(f".//{{{self.dmn_ns}}}rule"):
                    rule_data = {
                        "id": rule.get("id", ""),
                        "inputs": [],
                        "outputs": []
                    }
                    
                    for inp_entry in rule.findall(f".//{{{self.dmn_ns}}}inputEntry"):
                        text = inp_entry.find(f".//{{{self.dmn_ns}}}text")
                        rule_data["inputs"].append(text.text if text is not None else "")
                    
                    for out_entry in rule.findall(f".//{{{self.dmn_ns}}}outputEntry"):
                        text = out_entry.find(f".//{{{self.dmn_ns}}}text")
                        rule_data["outputs"].append(text.text if text is not None else "")
                    
                    dec_data["rules"].append(rule_data)
            
            decisions.append(dec_data)
        
        return {
            "type": "DMN",
            "inputData": input_data,
            "decisions": decisions
        }
    
    def generate_clinical_flowchart(self, rules_data: List[Dict], module_name: str, output_path: str) -> None:
        """Generate a comprehensive clinical decision flowchart."""
        if plt is None or nx is None:
            raise ImportError("matplotlib and networkx required for flowchart generation")
        
        # Create a figure with subplots for better organization
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))
        ax.set_title(f"Clinical Decision Workflow: {module_name.title()}", 
                    fontsize=20, fontweight='bold', pad=30)
        
        # Sort rules by priority for logical flow
        sorted_rules = sorted(rules_data, key=lambda x: x.get('priority', 0), reverse=True)
        
        # Calculate layout positions manually for better readability
        y_positions = {}
        x_center = 0.5
        
        # Start position
        start_y = 0.95
        y_positions['start'] = (x_center, start_y)
        
        # Position rules vertically with spacing
        rule_spacing = 0.12
        current_y = start_y - 0.08
        
        # Group rules by priority for visual separation
        priority_groups = {}
        for rule in sorted_rules:
            priority = rule.get('priority', 0)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(rule)
        
        rule_positions = {}
        decision_boxes = []
        
        for priority in sorted(priority_groups.keys(), reverse=True):
            if priority_groups[priority]:
                # Add priority group header
                current_y -= 0.05
                
                for rule in priority_groups[priority]:
                    rule_id = rule.get('rule_id', 'unknown')
                    rule_positions[rule_id] = (x_center, current_y)
                    
                    # Create detailed decision box info
                    when_conditions = rule.get('when', [])
                    then_clause = rule.get('then', {})
                    
                    # Parse conditions into readable text
                    condition_texts = []
                    for condition in when_conditions:
                        if isinstance(condition, dict):
                            if 'obs' in condition:
                                op_text = {
                                    'ge': '≥', 'gt': '>', 'le': '≤', 'lt': '<', 'eq': '='
                                }.get(condition['op'], condition['op'])
                                condition_texts.append(f"{condition['obs']} {op_text} {condition['value']}")
                            elif 'sym' in condition:
                                if condition['eq']:
                                    condition_texts.append(f"Has {condition['sym'].replace('_', ' ')}")
                                else:
                                    condition_texts.append(f"No {condition['sym'].replace('_', ' ')}")
                    
                    # Parse actions
                    actions = []
                    if then_clause.get('propose_triage'):
                        actions.append(f"Refer to {then_clause['propose_triage']}")
                    if then_clause.get('set_flags'):
                        for flag in then_clause['set_flags']:
                            actions.append(f"Flag: {flag.replace('.', ' ').title()}")
                    if then_clause.get('reasons'):
                        for reason in then_clause['reasons']:
                            actions.append(f"Reason: {reason.replace('.', ' ').title()}")
                    
                    decision_boxes.append({
                        'id': rule_id,
                        'position': (x_center, current_y),
                        'conditions': condition_texts,
                        'actions': actions,
                        'priority': priority,
                        'guideline_ref': then_clause.get('guideline_ref', '')
                    })
                    
                    current_y -= rule_spacing
        
        # End positions for different outcomes
        end_y = current_y - 0.05
        end_positions = {
            'hospital': (0.2, end_y),
            'clinic': (0.5, end_y), 
            'home': (0.8, end_y)
        }
        
        # Clear the axes
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Draw start node
        start_circle = plt.Circle(y_positions['start'], 0.03, 
                                color='lightgreen', alpha=0.8, zorder=3)
        ax.add_patch(start_circle)
        ax.text(y_positions['start'][0], y_positions['start'][1], 'START\nPatient\nAssessment', 
               ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
        
        # Draw decision boxes with detailed information
        for box in decision_boxes:
            x, y = box['position']
            
            # Determine box color based on priority
            if box['priority'] >= 100:
                box_color = '#ffcccc'  # Light red for high priority/danger signs
                border_color = 'red'
            elif box['priority'] >= 50:
                box_color = '#ffffcc'  # Light yellow for medium priority  
                border_color = 'orange'
            else:
                box_color = '#ccffcc'  # Light green for routine checks
                border_color = 'green'
            
            # Create detailed text for the box
            title = f"{box['id']} (Priority: {box['priority']})"
            
            condition_text = "IF:\n" + "\n".join([f"• {cond}" for cond in box['conditions']]) if box['conditions'] else "IF: Always"
            
            action_text = "THEN:\n" + "\n".join([f"• {action}" for action in box['actions']]) if box['actions'] else "THEN: No action"
            
            full_text = f"{title}\n\n{condition_text}\n\n{action_text}"
            
            # Calculate box size based on text
            box_height = 0.08 + len(box['conditions']) * 0.01 + len(box['actions']) * 0.01
            box_width = 0.25
            
            # Draw box
            rect = plt.Rectangle((x - box_width/2, y - box_height/2), 
                               box_width, box_height,
                               facecolor=box_color, edgecolor=border_color,
                               linewidth=2, alpha=0.9, zorder=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x, y, full_text, ha='center', va='center', 
                   fontsize=8, fontweight='normal', zorder=4,
                   bbox=dict(boxstyle="round,pad=0.01", facecolor='white', alpha=0.8))
        
        # Draw end nodes
        for outcome, (x, y) in end_positions.items():
            color_map = {
                'hospital': '#ff6b6b',  # Red
                'clinic': '#ffa726',    # Orange
                'home': '#66bb6a'       # Green
            }
            
            end_circle = plt.Circle((x, y), 0.04, 
                                  color=color_map[outcome], alpha=0.9, zorder=3)
            ax.add_patch(end_circle)
            ax.text(x, y, f'{outcome.upper()}\nReferral', 
                   ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
        
        # Draw flow arrows
        # From start to first decision
        if decision_boxes:
            first_box = decision_boxes[0]
            ax.annotate('', xy=(first_box['position'][0], first_box['position'][1] + 0.04),
                       xytext=(y_positions['start'][0], y_positions['start'][1] - 0.03),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Between decision boxes
        for i in range(len(decision_boxes) - 1):
            current = decision_boxes[i]
            next_box = decision_boxes[i + 1]
            
            ax.annotate('', xy=(next_box['position'][0], next_box['position'][1] + 0.04),
                       xytext=(current['position'][0], current['position'][1] - 0.04),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        
        # From last decision to outcomes
        if decision_boxes:
            last_box = decision_boxes[-1]
            for outcome, (x, y) in end_positions.items():
                ax.annotate('', xy=(x, y + 0.04),
                           xytext=(last_box['position'][0], last_box['position'][1] - 0.04),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))
        
        # Add legend
        legend_y = 0.15
        ax.text(0.05, legend_y, "PRIORITY LEVELS:", fontsize=12, fontweight='bold')
        ax.text(0.05, legend_y - 0.03, "🔴 High Priority (≥100): Danger signs, immediate action required", fontsize=9)
        ax.text(0.05, legend_y - 0.06, "🟡 Medium Priority (50-99): Important clinical findings", fontsize=9) 
        ax.text(0.05, legend_y - 0.09, "🟢 Routine Priority (<50): Standard assessments", fontsize=9)
        
        # Add reference information
        ax.text(0.05, 0.03, "Generated from WHO CHW Guidelines", fontsize=8, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_bpmn_flowchart(self, bpmn_path: str, output_path: str) -> None:
        """Generate a technical BPMN flowchart (legacy method)."""
        if plt is None or nx is None:
            raise ImportError("matplotlib and networkx required for flowchart generation")
        
        json_data = self.parse_bpmn_to_json(bpmn_path)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges from BPMN
        for process in json_data["processes"]:
            elements = process["elements"]
            
            # Add nodes
            for elem in elements:
                if elem["type"] != "sequenceFlow":
                    G.add_node(elem["id"], 
                              name=elem["name"] or elem["id"], 
                              type=elem["type"])
            
            # Add edges
            for elem in elements:
                if elem["type"] == "sequenceFlow":
                    source = elem.get("sourceRef")
                    target = elem.get("targetRef")
                    if source and target:
                        G.add_edge(source, target, name=elem["name"])
        
        # Create flowchart
        plt.figure(figsize=(16, 12))
        plt.title("Technical BPMN Process Flow", fontsize=16, fontweight='bold')
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Draw nodes with different shapes based on type
        node_colors = {
            'startEvent': 'lightgreen',
            'endEvent': 'lightcoral',
            'task': 'lightblue',
            'exclusiveGateway': 'yellow'
        }
        
        for node_type in node_colors:
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=node_colors[node_type],
                                     node_size=2000, alpha=0.8)
        
        # Draw other nodes in default color
        other_nodes = [n for n, d in G.nodes(data=True) 
                      if d.get('type') not in node_colors]
        if other_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                                 node_color='lightgray', node_size=2000, alpha=0.8)
        
        # Draw edges with labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=25, alpha=0.7, width=2)
        
        # Draw node labels
        labels = {n: d.get('name', n)[:20] + ('...' if len(d.get('name', n)) > 20 else '') 
                 for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): d.get('name', '')[:15] for u, v, d in G.edges(data=True) if d.get('name')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_dmn_flowchart(self, dmn_path: str, output_path: str) -> None:
        """Generate a flowchart from DMN XML."""
        if plt is None or nx is None:
            raise ImportError("matplotlib and networkx required for flowchart generation")
        
        json_data = self.parse_dmn_to_json(dmn_path)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add input data nodes
        for inp in json_data["inputData"]:
            G.add_node(inp["id"], name=inp["name"] or inp["id"], type="inputData")
        
        # Add decision nodes and their requirements
        for decision in json_data["decisions"]:
            G.add_node(decision["id"], 
                      name=decision["name"] or decision["id"], 
                      type="decision")
            
            # Add edges from inputs to decisions
            for req in decision["requirements"]:
                if req in [inp["id"] for inp in json_data["inputData"]]:
                    G.add_edge(req, decision["id"])
        
        # Create flowchart
        plt.figure(figsize=(10, 8))
        plt.title("DMN Decision Requirements Diagram", fontsize=16, fontweight='bold')
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Draw input data nodes (parallelograms approximated as ellipses)
        input_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'inputData']
        if input_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                                 node_color='lightblue', node_shape='s',
                                 node_size=2000, alpha=0.8)
        
        # Draw decision nodes (rectangles)
        decision_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'decision']
        if decision_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=decision_nodes,
                                 node_color='lightgreen', node_shape='s',
                                 node_size=2000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        labels = {n: d.get('name', n)[:12] + ('...' if len(d.get('name', n)) > 12 else '')
                 for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_json_format(self, xml_path: str, output_path: str, format_type: str) -> None:
        """Save BPMN or DMN in readable JSON format."""
        if format_type.lower() == "bpmn":
            json_data = self.parse_bpmn_to_json(xml_path)
        elif format_type.lower() == "dmn":
            json_data = self.parse_dmn_to_json(xml_path)
        else:
            raise ValueError("format_type must be 'bpmn' or 'dmn'")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
