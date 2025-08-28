from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple

from graphviz import Digraph


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


def render_drd(dmn_path: str, out_path: str) -> None:
    inputs, decisions, edges = _parse_dmn(dmn_path)
    dot = Digraph("DRD", format=out_path.split(".")[-1])
    dot.attr("graph", rankdir="LR")
    for i in inputs:
        dot.node(i, i, shape="parallelogram")
    for d in decisions:
        dot.node(d, d, shape="rectangle")
    for src, dst in edges:
        dot.edge(src, dst)

    try:
        dot.render(filename=out_path, cleanup=True)
    except Exception:
        dot.save(filename=out_path.rsplit(".", 1)[0] + ".dot")
        sys.exit(2)

