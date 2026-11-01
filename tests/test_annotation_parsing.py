from __future__ import annotations

import json
from pathlib import Path

from src.data.annotation_parsers import parse_labelme_json, parse_voc_xml, parse_yolo_txt


def test_parse_labelme_rectangle(tmp_path: Path) -> None:
    payload = {
        "version": "2.4.4",
        "shapes": [
            {
                "label": "spot",
                "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
                "shape_type": "rectangle",
            }
        ],
        "imagePath": "example.jpg",
        "imageWidth": 100,
        "imageHeight": 80,
    }
    path = tmp_path / "example.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = parse_labelme_json(path)
    assert parsed.raw_ok
    assert parsed.image_filename == "example.jpg"
    assert parsed.recorded_width == 100
    assert len(parsed.boxes) == 1
    box = parsed.boxes[0]
    assert box.class_raw == "spot"
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (10, 20, 30, 40)


def test_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    parsed = parse_labelme_json(path)
    assert parsed.raw_ok is False
    assert any(issue.code == "malformed_annotation" for issue in parsed.issues)


def test_parse_voc_xml(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
    <annotation>
      <filename>example.jpg</filename>
      <size><width>100</width><height>80</height><depth>3</depth></size>
      <object>
        <name>scar</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>11</xmax><ymax>22</ymax></bndbox>
      </object>
    </annotation>
    """
    path = tmp_path / "example.xml"
    path.write_text(xml, encoding="utf-8")
    parsed = parse_voc_xml(path)
    assert parsed.raw_ok
    assert parsed.boxes[0].class_raw == "scar"
    assert parsed.boxes[0].xmax == 11


def test_malformed_xml(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text("<annotation><object></annotation>", encoding="utf-8")
    parsed = parse_voc_xml(path)
    assert parsed.raw_ok is False
    assert any(issue.code == "malformed_annotation" for issue in parsed.issues)


def test_missing_voc_coordinates(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
    <annotation>
      <filename>example.jpg</filename>
      <size><width>10</width><height>10</height></size>
      <object>
        <name>spot</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin></bndbox>
      </object>
    </annotation>
    """
    path = tmp_path / "missing.xml"
    path.write_text(xml, encoding="utf-8")
    parsed = parse_voc_xml(path)
    assert parsed.boxes == []
    assert any(issue.code == "missing_coordinate" for issue in parsed.issues)


def test_parse_yolo_txt(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    parsed = parse_yolo_txt(path, image_width=100, image_height=50)
    assert len(parsed.boxes) == 1
    box = parsed.boxes[0]
    assert box.class_id == 2
    assert abs(box.xmin - 40) < 1e-6
    assert abs(box.ymin - 15) < 1e-6
    assert abs(box.xmax - 60) < 1e-6
    assert abs(box.ymax - 35) < 1e-6


def test_yolo_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("2 0.5 0.5\n", encoding="utf-8")
    parsed = parse_yolo_txt(path, 10, 10)
    assert parsed.boxes == []
    assert any(issue.code == "malformed_annotation" for issue in parsed.issues)


def test_empty_json_shapes(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"shapes": [], "imageWidth": 10, "imageHeight": 10}), encoding="utf-8")
    parsed = parse_labelme_json(path)
    assert parsed.boxes == []
    assert any(issue.code == "missing_objects" for issue in parsed.issues)
