# AgriData-Agent

**Autonomous Data-Engineering & Visual-QA Agent for Multi-Class Agricultural Object Detection Datasets**

AgriData-Agent is a modular, tool-calling AI agent and CLI toolkit built to automate dataset auditing, bounding-box anomaly detection, multi-label stratified splitting, and headless visual quality assurance across **YOLO**, **Pascal VOC**, and **COCO** datasets. Designed for high-density, occluded field imagery (such as multi-species weed detection benchmarks like **MH-Weed16**), it runs seamlessly on CPU/headless Linux servers and scales to GPU environments for vision-embedding deduplication and local VLM verification.

## Core Capabilities

- **Multi-Format Parsing (`parsers.py`):** Unified ingestion for YOLO (`.txt`), Pascal VOC (`.xml`), and COCO (`.json`) annotations with automatic coordinate normalization.
- **Dataset Health & Geometry Auditing (`validator.py`):** Detects missing/orphan label pairs, quantifies multi-class skew across rare species, and flags out-of-bounds coordinates, zero-area boxes, extreme aspect ratios, and occlusion-induced micro-boxes.
- **Rare-Class-First Multi-Label Stratification (`splitter.py`):** Allocates images containing minority weed species first across `train.txt`, `val.txt`, and `test.txt` manifests to prevent rare-class dropout and split leakage.
- **Headless QA Contact Sheets (`visualizer.py`):** Renders color-coded bounding boxes onto anomalous or high-density frames and stitches them into a single inspection grid (`qa_contact_sheet.jpg`) without requiring a display server.
- **Flexible Agent Execution (`agent.py`):** Supports any OpenAI-compatible endpoint (cloud APIs or local GPU-hosted **Ollama** / **vLLM** models) as well as a deterministic `--offline` execution mode.

## Installation & Quickstart

    git clone [https://github.com/SUBINOY551/agridata-agent.git](https://github.com/SUBINOY551/agridata-agent.git)
    cd agridata-agent
    pip install -e ".[dev]"

Run deterministic dataset audit:

    agridata audit --dataset ./path/to/dataset --json-out audit_stats.json

Run full autonomous agent pipeline:

    agridata run --dataset ./path/to/dataset --output ./splits --report audit_report.md

Run unit test suite:

    pytest -v

## Active Development Roadmap (GPU Upgrade Phase)

- [ ] **GPU DINOv2 Near-Duplicate & Leakage Filter:** Batch-extract L2-normalized vision embeddings on CUDA (`timm`) to cluster sequential field frames before splitting.
- [ ] **Zero-Shot VLM Bounding-Box Verification:** Integrate local Vision-Language Models (Qwen2.5-VL / Florence-2) on GPU to flag mislabeled or heavily occluded weed bounding boxes.
- [ ] **Automated Albumentations Policy Generator:** Synthesize class-aware augmentation pipelines directly from the agent telemetry.
- [ ] **Interactive COCO/YOLO Cross-Format Auto-Repair:** Add safe bounding-box clamping and automated quarantine manifests for corrupted label rows.
