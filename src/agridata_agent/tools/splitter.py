from collections import Counter
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple
from agridata_agent.schemas import SplitSummary
from agridata_agent.tools.parsers import (
    IMG_EXTS,
    parse_coco_json,
    parse_voc_file,
    parse_yolo_file,
)


def create_stratified_split(
    dataset_dir: str,
    output_dir: str,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> SplitSummary:
    """Performs iterative multi-label stratified splitting (rare classes assigned first) across YOLO/VOC/COCO."""
    if round(sum(ratios), 4) != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    root = Path(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS}
    yolo_files = [p for p in root.rglob("*.txt") if p.name not in {"classes.txt", "train.txt", "val.txt", "test.txt"}]
    voc_files = list(root.rglob("*.xml"))
    coco_files = list(root.rglob("*.json"))

    img_classes: Dict[str, Set[str]] = {stem: set() for stem in images}
    global_counts: Counter[str] = Counter()

    for txt_path in yolo_files:
        if txt_path.stem in img_classes:
            recs, _ = parse_yolo_file(txt_path)
            for r in recs:
                img_classes[txt_path.stem].add(r.class_label)
                global_counts[r.class_label] += 1

    for xml_path in voc_files:
        if xml_path.stem in img_classes:
            recs, _ = parse_voc_file(xml_path)
            for r in recs:
                img_classes[xml_path.stem].add(r.class_label)
                global_counts[r.class_label] += 1

    for json_path in coco_files:
        grouped, _ = parse_coco_json(json_path)
        for stem, recs in grouped.items():
            if stem in img_classes:
                for r in recs:
                    img_classes[stem].add(r.class_label)
                    global_counts[r.class_label] += 1

    # Shuffle deterministically, then sort so images containing the rarest species are assigned first
    ordered_stems = sorted(images.keys())
    rng.shuffle(ordered_stems)
    ordered_stems.sort(
        key=lambda s: min((global_counts[c] for c in img_classes[s]), default=10**9)
    )

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_class_counts: Dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    target_ratios = {"train": ratios[0], "val": ratios[1], "test": ratios[2]}

    for stem in ordered_stems:
        present = img_classes[stem]
        if not present:
            chosen = min(splits.keys(), key=lambda k: len(splits[k]) / target_ratios[k])
        else:
            rarest_cls = min(present, key=lambda c: global_counts[c])
            chosen = min(
                splits.keys(),
                key=lambda k: (
                    split_class_counts[k][rarest_cls] / target_ratios[k],
                    len(splits[k]) / target_ratios[k],
                ),
            )
        splits[chosen].append(str(images[stem].resolve()))
        for c in present:
            split_class_counts[chosen][c] += 1

    for split_name, paths in splits.items():
        (out / f"{split_name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    return SplitSummary(
        output_dir=str(out),
        train_images=len(splits["train"]),
        val_images=len(splits["val"]),
        test_images=len(splits["test"]),
        class_distribution_by_split={
            k: dict(sorted(v.items())) for k, v in split_class_counts.items()
        },
    )
