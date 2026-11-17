import json
import os
from typing import Any, Callable, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

from agridata_agent.tools.splitter import create_stratified_split
from agridata_agent.tools.validator import audit_dataset
from agridata_agent.tools.visualizer import render_qa_contact_sheet

load_dotenv()

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "audit_dataset",
            "description": (
                "Recursively audit a YOLO, Pascal VOC, or COCO dataset for missing/orphan labels, "
                "multi-class skew/imbalance, and bounding-box geometry anomalies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string", "description": "Root path of the dataset"},
                    "min_box_area_norm": {"type": "number", "description": "Minimum normalized box area threshold"},
                },
                "required": ["dataset_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_near_duplicates_gpu",
            "description": (
                "Use GPU vision embeddings (DINOv2) to detect near-duplicate sequential field images "
                "that cause train/validation data leakage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string", "description": "Root path of the dataset"},
                    "similarity_threshold": {"type": "number", "description": "Cosine similarity threshold (default 0.96)"},
                },
                "required": ["dataset_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_qa_contact_sheet",
            "description": "Render bounding boxes onto anomalous or dense images into a headless QA grid image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string"},
                    "output_path": {"type": "string"},
                    "image_paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dataset_dir", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_stratified_split",
            "description": "Generate rare-class-balanced multi-label stratified train/val/test manifests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                },
                "required": ["dataset_dir", "output_dir"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are AgriData-Agent, an autonomous Computer Vision Data Engineering assistant.
Follow this workflow:
1. Call `audit_dataset` first to inspect formats, pairing errors, class skew, and box anomalies.
2. Call `find_near_duplicates_gpu` to check for near-duplicate images that risk split leakage.
3. Call `render_qa_contact_sheet` to generate visual evidence of flagged or high-density frames.
4. Call `create_stratified_split` to produce leak-free, rare-class-balanced train/val/test manifests.
5. Conclude with a concise Markdown Engineering Audit Report."""


def _run_gpu_dedup_safe(**kw) -> str:
    try:
        from agridata_agent.tools.gpu_dedup import find_near_duplicates_gpu
        clusters = find_near_duplicates_gpu(**kw)
        return json.dumps(
            {
                "duplicate_clusters_found": len(clusters),
                "clusters": [c.model_dump() for c in clusters[:25]],
            },
            indent=2,
        )
    except ImportError:
        return json.dumps({"warning": "GPU extras not installed (pip install -e .[gpu]). Skipping embedding dedup."})


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    handlers: Dict[str, Callable[..., Any]] = {
        "audit_dataset": lambda **kw: audit_dataset(**kw).model_dump_json(indent=2),
        "find_near_duplicates_gpu": _run_gpu_dedup_safe,
        "render_qa_contact_sheet": lambda **kw: json.dumps({"contact_sheet_path": render_qa_contact_sheet(**kw)}),
        "create_stratified_split": lambda **kw: create_stratified_split(**kw).model_dump_json(indent=2),
    }
    if name not in handlers:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return handlers[name](**args)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def run_deterministic_pipeline(dataset_dir: str, output_dir: str) -> str:
    audit = audit_dataset(dataset_dir)
    dedup_raw = json.loads(_run_gpu_dedup_safe(dataset_dir=dataset_dir))
    flagged_imgs = list(dict.fromkeys(a.image_path for a in audit.anomalies if a.image_path != "MISSING"))
    qa_path = render_qa_contact_sheet(
        dataset_dir=dataset_dir,
        output_path=f"{output_dir}/qa_contact_sheet.jpg",
        image_paths=flagged_imgs[:9] if flagged_imgs else None,
    )
    splits = create_stratified_split(dataset_dir=dataset_dir, output_dir=output_dir)

    return f"""# AgriData-Agent Engineering Audit Report

## 1. Dataset Integrity & Format Summary
- **Dataset Root:** `{audit.dataset_root}`
- **Detected Formats:** {", ".join(audit.detected_formats) or "None"}
- **Total Images:** {audit.total_images}
- **Total Annotation Files:** {audit.total_annotation_files}
- **Missing / Orphan Annotations:** {len(audit.missing_annotations)} / {len(audit.orphan_annotations)}

## 2. Class Distribution & Imbalance Diagnostics
- **Max/Min Imbalance Ratio:** `{audit.imbalance_ratio}x`
- **Rare Classes (<15% of majority class):** {", ".join(audit.rare_classes) if audit.rare_classes else "None"}
- **Per-Class Instance Counts:** `{json.dumps(audit.class_counts)}`

## 3. GPU Near-Duplicate & Leakage Check
- **Duplicate Clusters Detected:** {dedup_raw.get("duplicate_clusters_found", "Skipped (CPU-only install)")}

## 4. Bounding-Box Geometry & Visual QA
- **Total Geometry/Format Anomalies Detected:** {audit.anomalies_found}
- **QA Contact Sheet Generated:** `{qa_path}`

## 5. Multi-Label Stratified Splits
- **Manifest Directory:** `{splits.output_dir}`
- **Train / Val / Test Counts:** `{splits.train_images}` / `{splits.val_images}` / `{splits.test_images}`
"""


def run_agent(
    dataset_dir: str,
    output_dir: str = "./splits",
    objective: str = "Audit dataset health, check GPU near-duplicates, render QA contact sheet, and build stratified splits.",
    max_steps: int = 8,
    offline: bool = False,
) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if offline or not api_key or api_key == "your_api_key_here":
        return run_deterministic_pipeline(dataset_dir=dataset_dir, output_dir=output_dir)

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Dataset directory: {dataset_dir}\nOutput directory: {output_dir}\nObjective: {objective}",
        },
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or "Agent finished without text output."

        for tc in msg.tool_calls:
            tool_output = _dispatch_tool(tc.function.name, json.loads(tc.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})

    return "Agent reached maximum reasoning steps before completing report."
