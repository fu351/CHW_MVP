from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple


DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"
ns = {"dmn": DMN_NS}


def _parse_dmn(dmn_path: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    tree = ET.parse(dmn_path)
    root = tree.getroot()
    inputs = [el.attrib.get("id", el.attrib.get("name", "")) for el in root.findall("dmn:inputData", ns)]
    decisions = [el.attrib.get("id", el.attrib.get("name", "")) for el in root.findall("dmn:decision", ns)]
    edges: List[Tuple[str, str]] = []
    for dec in root.findall("dmn:decision", ns):
        dec_id = dec.attrib.get("id", dec.attrib.get("name", "decision"))
        for ir in dec.findall("dmn:informationRequirement", ns):
            req = ir.find("dmn:requiredInput", ns)
            if req is not None:
                href = req.attrib.get("href", "")
                if href.startswith("#"):
                    edges.append((href[1:], dec_id))
    return inputs, decisions, edges


essential_shapes = {
    "input": "parallelogram",
    "decision": "rectangle",
}


def render_drd(dmn_path: str, out_path: str) -> None:
    inputs, decisions, edges = _parse_dmn(dmn_path)
    try:
        from graphviz import Digraph  # lazy import
    except Exception:
        # If graphviz is not available, save DOT and exit 2
        lines = ["digraph DRD {", "rankdir=LR;"]
        for i in inputs:
            lines.append(f'  "{i}" [shape=parallelogram];')
        for d in decisions:
            lines.append(f'  "{d}" [shape=rectangle];')
        for src, dst in edges:
            lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        dot_path = out_path.rsplit(".", 1)[0] + ".dot"
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        sys.exit(2)

    dot = Digraph("DRD", format=out_path.split(".")[-1])
    dot.attr("graph", rankdir="LR")
    for i in inputs:
        dot.node(i, i, shape=essential_shapes["input"])
    for d in decisions:
        dot.node(d, d, shape=essential_shapes["decision"])
    for src, dst in edges:
        dot.edge(src, dst)

    try:
        dot.render(filename=out_path, cleanup=True)
    except Exception:
        dot.save(filename=out_path.rsplit(".", 1)[0] + ".dot")
        sys.exit(2)
