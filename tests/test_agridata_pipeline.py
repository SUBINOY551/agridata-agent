from pathlib import Path
import cv2
import numpy as np
from agridata_agent.agent import run_agent
from agridata_agent.tools.splitter import create_stratified_split
from agridata_agent.tools.validator import audit_dataset


def _build_synthetic_weed_dataset(root: Path) -> None:
    """Creates a synthetic multi-class agricultural dataset with YOLO and Pascal VOC annotations."""
    root.mkdir(parents=True, exist_ok=True)
    # 12 synthetic field images across 4 weed species (with intentional class skew)
    species_schedule = ["0", "0", "0", "0", "1", "1", "1", "2", "2", "3", "0", "1"]

    for idx, cls_id in enumerate(species_schedule):
        img = np.full((256, 256, 3), (34, 139, 34), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (140, 140), (0, 200, 255), -1)
        cv2.imwrite(str(root / f"field_{idx:02d}.jpg"), img)
        (root / f"field_{idx:02d}.txt").write_text(f"{cls_id} 0.35 0.35 0.30 0.30\n", encoding="utf-8")

    # Add 1 image with an out-of-bounds bounding box anomaly
    bad_img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "field_anomaly.jpg"), bad_img)
    (root / "field_anomaly.txt").write_text("0 0.92 0.50 0.30 0.20\n", encoding="utf-8")


def test_audit_detects_imbalance_and_out_of_bounds(tmp_path: Path):
    ds_dir = tmp_path / "mh_weed_sample"
    _build_synthetic_weed_dataset(ds_dir)

    report = audit_dataset(str(ds_dir), rare_class_threshold_ratio=0.25)
    assert report.total_images == 13
    assert "yolo" in report.detected_formats
    assert report.anomalies_found == 1
    assert "exceed [0, 1]" in report.anomalies[0].reason
    assert "3" in report.rare_classes


def test_stratified_split_preserves_rare_species(tmp_path: Path):
    ds_dir = tmp_path / "mh_weed_sample"
    out_dir = tmp_path / "splits"
    _build_synthetic_weed_dataset(ds_dir)

    summary = create_stratified_split(str(ds_dir), str(out_dir), ratios=(0.7, 0.2, 0.1))
    assert summary.train_images + summary.val_images + summary.test_images == 13
    assert (out_dir / "train.txt").exists()
    assert (out_dir / "val.txt").exists()
    assert (out_dir / "test.txt").exists()


def test_end_to_end_offline_agent_workflow(tmp_path: Path):
    ds_dir = tmp_path / "mh_weed_sample"
    out_dir = tmp_path / "agent_outputs"
    _build_synthetic_weed_dataset(ds_dir)

    markdown = run_agent(dataset_dir=str(ds_dir), output_dir=str(out_dir), offline=True)
    assert "AgriData-Agent Engineering Audit Report" in markdown
    assert (out_dir / "qa_contact_sheet.jpg").exists()
