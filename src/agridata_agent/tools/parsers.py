import json
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from agridata_agent.schemas import AnnotationFormat, BoundingBoxRecord

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_yolo_file(txt_path: Path) -> Tuple[List[BoundingBoxRecord], List[str]]:
    """Parses normalized YOLO .txt annotations into BoundingBoxRecord objects."""
    records: List[BoundingBoxRecord] = []
    errors: List[str] = []

    for idx, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"Line {idx}: malformed YOLO row ({len(parts)} columns)")
            continue
        try:
            cls_label = str(int(float(parts[0])))
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            errors.append(f"Line {idx}: non-numeric coordinate values")
            continue

        x1, y1 = xc - w / 2.0, yc - h / 2.0
        x2, y2 = xc + w / 2.0, yc + h / 2.0
        records.append(
            BoundingBoxRecord(
                class_label=cls_label,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                normalized=True,
            )
        )
    return records, errors


def parse_voc_file(xml_path: Path) -> Tuple[List[BoundingBoxRecord], List[str]]:
    """Parses Pascal VOC .xml annotations, normalizing coordinates if image <size> is present."""
    records: List[BoundingBoxRecord] = []
    errors: List[str] = []

    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as exc:
        return [], [f"XML ParseError: {exc}"]

    size_node = root.find("size")
    img_w = float(size_node.findtext("width", default="0")) if size_node is not None else 0.0
    img_h = float(size_node.findtext("height", default="0")) if size_node is not None else 0.0
    can_normalize = img_w > 0 and img_h > 0

    for idx, obj in enumerate(root.findall("object"), start=1):
        name = obj.findtext("name", default="UNKNOWN").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            errors.append(f"Object {idx}: missing <bndbox>")
            continue
        try:
            x1 = float(bnd.findtext("xmin", default="0"))
            y1 = float(bnd.findtext("ymin", default="0"))
            x2 = float(bnd.findtext("xmax", default="0"))
            y2 = float(bnd.findtext("ymax", default="0"))
        except ValueError:
            errors.append(f"Object {idx}: non-numeric bndbox coordinates")
            continue

        if can_normalize:
            x1, x2 = x1 / img_w, x2 / img_w
            y1, y2 = y1 / img_h, y2 / img_h

        records.append(
            BoundingBoxRecord(
                class_label=name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                normalized=can_normalize,
            )
        )
    return records, errors


def parse_coco_json(json_path: Path) -> Tuple[dict[str, List[BoundingBoxRecord]], List[str]]:
    """Parses a COCO dataset JSON into a mapping of image_stem -> BoundingBoxRecords."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not (isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()):
        return {}, ["Not a valid COCO schema"]

    categories = {c["id"]: str(c.get("name", c["id"])) for c in data["categories"]}
    images_by_id = {im["id"]: im for im in data["images"]}
    grouped: dict[str, List[BoundingBoxRecord]] = {
        Path(im["file_name"]).stem: [] for im in data["images"] if "file_name" in im
    }
    errors: List[str] = []

    for idx, ann in enumerate(data["annotations"], start=1):
        im = images_by_id.get(ann.get("image_id"))
        if im is None:
            errors.append(f"Annotation {idx}: references missing image_id {ann.get('image_id')}")
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"Annotation {idx}: invalid bbox length {len(bbox)}")
            continue

        x, y, w, h = map(float, bbox)
        img_w = float(im.get("width", 0))
        img_h = float(im.get("height", 0))
        can_norm = img_w > 0 and img_h > 0

        x1, y1, x2, y2 = x, y, x + w, y + h
        if can_norm:
            x1, x2 = x1 / img_w, x2 / img_w
            y1, y2 = y1 / img_h, y2 / img_h

        stem = Path(im["file_name"]).stem
        grouped.setdefault(stem, []).append(
            BoundingBoxRecord(
                class_label=categories.get(ann.get("category_id"), str(ann.get("category_id"))),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                normalized=can_norm,
            )
        )
    return grouped, errors


def detect_format(path: Path) -> AnnotationFormat:
    ext = path.suffix.lower()
    if ext == ".txt":
        return "yolo"
    if ext == ".xml":
        return "voc"
    if ext == ".json":
        return "coco"
    return "unknown"
