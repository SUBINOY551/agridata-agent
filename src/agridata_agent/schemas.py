from typing import Dict, List, Literal, Tuple
from pydantic import BaseModel, Field

AnnotationFormat = Literal["yolo", "voc", "coco", "via", "unknown"]


class BoundingBoxRecord(BaseModel):
    class_label: str
    # Normalized coordinates [x1, y1, x2, y2] in [0, 1] when image size is known,
    # or raw pixel coordinates when unnormalized.
    x1: float
    y1: float
    x2: float
    y2: float
    normalized: bool = True


class BBoxAnomaly(BaseModel):
    image_path: str
    annotation_path: str
    annotation_format: AnnotationFormat
    record_index: int
    class_label: str
    reason: str
    coords: Tuple[float, float, float, float]


class DatasetAuditReport(BaseModel):
    dataset_root: str
    detected_formats: List[str]
    total_images: int
    total_annotation_files: int
    missing_annotations: List[str] = Field(default_factory=list)
    orphan_annotations: List[str] = Field(default_factory=list)
    class_counts: Dict[str, int] = Field(default_factory=dict)
    imbalance_ratio: float
    rare_classes: List[str] = Field(default_factory=list)
    anomalies_found: int
    anomalies: List[BBoxAnomaly] = Field(default_factory=list)


class SplitSummary(BaseModel):
    output_dir: str
    train_images: int
    val_images: int
    test_images: int
    class_distribution_by_split: Dict[str, Dict[str, int]]
