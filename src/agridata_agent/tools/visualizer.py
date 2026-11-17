from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from agridata_agent.schemas import BoundingBoxRecord
from agridata_agent.tools.parsers import (
    IMG_EXTS,
    parse_coco_json,
    parse_voc_file,
    parse_yolo_file,
)


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministic high-contrast BGR color per weed class."""
    h = abs(hash(label))
    return (int(50 + (h % 205)), int(50 + ((h // 256) % 205)), int(50 + ((h // 65536) % 205)))


def render_qa_contact_sheet(
    dataset_dir: str,
    output_path: str = "qa_contact_sheet.jpg",
    image_paths: Optional[List[str]] = None,
    max_images: int = 9,
    cols: int = 3,
    tile_size: Tuple[int, int] = (512, 384),
) -> str:
    """Renders bounding boxes onto selected/crowded images and tiles them into a headless QA grid."""
    root = Path(dataset_dir)
    all_images = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not all_images:
        raise FileNotFoundError(f"No images found under {dataset_dir}")

    # Pre-index COCO JSONs if present
    coco_index: Dict[str, List[BoundingBoxRecord]] = {}
    for json_path in root.rglob("*.json"):
        grouped, _ = parse_coco_json(json_path)
        coco_index.update(grouped)

    # Load boxes for candidate images
    annotated_candidates: List[Tuple[Path, List[BoundingBoxRecord]]] = []
    target_paths = [Path(p) for p in image_paths] if image_paths else all_images

    for img_path in target_paths:
        if not img_path.exists():
            continue
        stem = img_path.stem
        recs: List[BoundingBoxRecord] = []

        txt_hits = [p for p in root.rglob(f"{stem}.txt") if p.name not in {"classes.txt", "train.txt", "val.txt", "test.txt"}]
        xml_hits = list(root.rglob(f"{stem}.xml"))

        if txt_hits:
            recs, _ = parse_yolo_file(txt_hits[0])
        elif xml_hits:
            recs, _ = parse_voc_file(xml_hits[0])
        elif stem in coco_index:
            recs = coco_index[stem]

        annotated_candidates.append((img_path, recs))

    # If no explicit list provided, prioritize highest object-density images for visual QA
    if not image_paths:
        annotated_candidates.sort(key=lambda item: len(item[1]), reverse=True)

    selected = annotated_candidates[:max_images]
    if not selected:
        raise ValueError("No valid images available to render QA contact sheet.")

    tiles: List[np.ndarray] = []
    tw, th = tile_size

    for img_path, recs in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        for box in recs:
            if box.normalized:
                x1, y1 = int(box.x1 * tw), int(box.y1 * th)
                x2, y2 = int(box.x2 * tw), int(box.y2 * th)
            else:
                sx, sy = tw / max(orig_w, 1), th / max(orig_h, 1)
                x1, y1 = int(box.x1 * sx), int(box.y1 * sy)
                x2, y2 = int(box.x2 * sx), int(box.y2 * sy)

            color = _color_for_label(box.class_label)
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                resized,
                box.class_label,
                (max(0, x1), max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            resized,
            f"{img_path.name} ({len(recs)} boxes)",
            (8, th - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(resized)

    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)
    return str(out_file.resolve())
