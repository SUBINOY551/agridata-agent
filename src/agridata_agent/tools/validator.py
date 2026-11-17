from collections import Counter
from pathlib import Path
from typing import Dict, List, Set
from agridata_agent.schemas import BBoxAnomaly, DatasetAuditReport
from agridata_agent.tools.parsers import (
    IMG_EXTS,
    parse_coco_json,
    parse_voc_file,
    parse_yolo_file,
)


def audit_dataset(
    dataset_dir: str,
    min_box_area_norm: float = 1e-5,
    max_aspect_ratio: float = 15.0,
    rare_class_threshold_ratio: float = 0.15,
) -> DatasetAuditReport:
    """Recursively audits YOLO, Pascal VOC, or COCO datasets for pairing, skew, and box geometry."""
    root = Path(dataset_dir)
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS}

    yolo_files = [p for p in root.rglob("*.txt") if p.name not in {"classes.txt", "train.txt", "val.txt", "test.txt"}]
    voc_files = list(root.rglob("*.xml"))
    coco_files = list(root.rglob("*.json"))

    detected_formats: Set[str] = set()
    annotations_by_stem: Dict[str, tuple[Path, str, list]] = {}
    anomalies: List[BBoxAnomaly] = []

    for txt_path in yolo_files:
        detected_formats.add("yolo")
        recs, errs = parse_yolo_file(txt_path)
        annotations_by_stem[txt_path.stem] = (txt_path, "yolo", recs)
        for err in errs:
            anomalies.append(
                BBoxAnomaly(
                    image_path=str(images.get(txt_path.stem, "MISSING")),
                    annotation_path=str(txt_path),
                    annotation_format="yolo",
                    record_index=0,
                    class_label="UNKNOWN",
                    reason=err,
                    coords=(0.0, 0.0, 0.0, 0.0),
                )
            )

    for xml_path in voc_files:
        detected_formats.add("voc")
        recs, errs = parse_voc_file(xml_path)
        annotations_by_stem[xml_path.stem] = (xml_path, "voc", recs)
        for err in errs:
            anomalies.append(
                BBoxAnomaly(
                    image_path=str(images.get(xml_path.stem, "MISSING")),
                    annotation_path=str(xml_path),
                    annotation_format="voc",
                    record_index=0,
                    class_label="UNKNOWN",
                    reason=err,
                    coords=(0.0, 0.0, 0.0, 0.0),
                )
            )

    for json_path in coco_files:
        grouped, errs = parse_coco_json(json_path)
        if grouped:
            detected_formats.add("coco")
            for stem, recs in grouped.items():
                annotations_by_stem[stem] = (json_path, "coco", recs)

    missing_annotations = sorted([str(images[s]) for s in (images.keys() - annotations_by_stem.keys())])
    orphan_annotations = sorted(
        [str(annotations_by_stem[s][0]) for s in (annotations_by_stem.keys() - images.keys())]
    )

    class_counts: Counter[str] = Counter()

    for stem, (ann_path, fmt, recs) in annotations_by_stem.items():
        img_str = str(images.get(stem, "MISSING"))
        for idx, box in enumerate(recs, start=1):
            class_counts[box.class_label] += 1
            w = box.x2 - box.x1
            h = box.y2 - box.y1

            if w <= 0 or h <= 0:
                reason = "Zero or inverted bounding box dimensions"
            elif box.normalized and (
                box.x1 < -1e-4 or box.y1 < -1e-4 or box.x2 > 1.0 + 1e-4 or box.y2 > 1.0 + 1e-4
            ):
                reason = "Normalized coordinates exceed [0, 1] image bounds"
            elif box.normalized and (w * h) < min_box_area_norm:
                reason = f"Micro-box area ({w * h:.2e}) likely caused by extreme occlusion"
            elif max(w / max(h, 1e-8), h / max(w, 1e-8)) > max_aspect_ratio:
                reason = f"Extreme aspect ratio (>{max_aspect_ratio}:1)"
            else:
                continue

            anomalies.append(
                BBoxAnomaly(
                    image_path=img_str,
                    annotation_path=str(ann_path),
                    annotation_format=fmt,  # type: ignore[arg-type]
                    record_index=idx,
                    class_label=box.class_label,
                    reason=reason,
                    coords=(round(box.x1, 5), round(box.y1, 5), round(box.x2, 5), round(box.y2, 5)),
                )
            )

    counts = list(class_counts.values())
    max_count = max(counts) if counts else 0
    imbalance_ratio = round(max_count / max(min(counts), 1), 2) if counts else 0.0
    rare_classes = sorted(
        [cls for cls, cnt in class_counts.items() if max_count > 0 and (cnt / max_count) < rare_class_threshold_ratio]
    )

    return DatasetAuditReport(
        dataset_root=str(root),
        detected_formats=sorted(detected_formats),
        total_images=len(images),
        total_annotation_files=len(yolo_files) + len(voc_files) + len(coco_files),
        missing_annotations=missing_annotations,
        orphan_annotations=orphan_annotations,
        class_counts=dict(sorted(class_counts.items())),
        imbalance_ratio=imbalance_ratio,
        rare_classes=rare_classes,
        anomalies_found=len(anomalies),
        anomalies=anomalies[:100],
    )
