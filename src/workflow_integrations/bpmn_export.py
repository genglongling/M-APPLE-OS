"""Export BPMN 2.0 XML for documentation-quality diagrams (optional)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
DC_NS = "http://www.omg.org/spec/DD/20100524/DC"
DI_NS = "http://www.omg.org/spec/DD/20100524/DI"


def _q(tag: str) -> str:
    return f"{{{BPMN_NS}}}{tag}"


def export_bpmn_diagram(workflow: dict[str, Any], out_path: str) -> str:
    """
    Emit BPMN 2.0 XML (process + minimal DI layout).

    For documentation and interchange; execution remains Python/ALAS by default.
    Reference: BPMN 2.0 documentation-quality diagrams [optional].
    """
    proc_id = workflow.get("name", "process").replace(" ", "_")
    steps = workflow.get("steps", [])

    root = ET.Element(_q("definitions"), attrib={"id": "alas-definitions"})
    root.set("xmlns:bpmn", BPMN_NS)
    root.set("xmlns:bpmndi", BPMNDI_NS)
    root.set("xmlns:dc", DC_NS)
    root.set("xmlns:di", DI_NS)

    process = ET.SubElement(root, _q("process"), id=proc_id, name=workflow.get("name", proc_id))

    start = ET.SubElement(process, _q("startEvent"), id="start")
    prev_flow_target = "start"
    x = 120

    for step in steps:
        sid = step["id"]
        task = ET.SubElement(
            process,
            _q("task"),
            id=sid,
            name=step.get("agent", sid),
        )
        doc = ET.SubElement(task, _q("documentation"))
        doc.text = f"kind={step.get('kind', 'agent')}; optional={step.get('optional', False)}"

        flow_id = f"flow_{prev_flow_target}_to_{sid}"
        ET.SubElement(
            process,
            _q("sequenceFlow"),
            id=flow_id,
            sourceRef=prev_flow_target,
            targetRef=sid,
        )
        prev_flow_target = sid
        x += 160

    end = ET.SubElement(process, _q("endEvent"), id="end")
    ET.SubElement(
        process,
        _q("sequenceFlow"),
        id=f"flow_{prev_flow_target}_to_end",
        sourceRef=prev_flow_target,
        targetRef="end",
    )

    bpmndi = ET.SubElement(root, f"{{{BPMNDI_NS}}}BPMNDiagram", id="BPMNDiagram_1")
    plane = ET.SubElement(
        bpmndi, f"{{{BPMNDI_NS}}}BPMNPlane", id="BPMNPlane_1", bpmnElement=proc_id
    )

    _di_shape(plane, "start", 80, 80, 36, 36)
    px = 180
    for step in steps:
        _di_shape(plane, step["id"], px, 68, 120, 60)
        px += 160
    _di_shape(plane, "end", px, 80, 36, 36)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    return out_path


def _di_shape(plane: ET.Element, element_id: str, x: int, y: int, w: int, h: int) -> None:
    shape = ET.SubElement(
        plane,
        f"{{{BPMNDI_NS}}}BPMNShape",
        id=f"{element_id}_di",
        bpmnElement=element_id,
    )
    bounds = ET.SubElement(shape, f"{{{DC_NS}}}Bounds")
    bounds.set("x", str(x))
    bounds.set("y", str(y))
    bounds.set("width", str(w))
    bounds.set("height", str(h))
