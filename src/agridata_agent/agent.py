import json
import os
from typing import Any, Callable, Dict, List, Optional
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
                    "min_box_area_norm": {
                        "type": "number",
                        "description": "Minimum normalized box area threshold (default 1e-5)",
                    },
                },
                "required": ["dataset_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_qa_contact_sheet",
            "description": (
                "Render bounding boxes onto anomalous or highest-density images and save a headless "
                "visual QA contact-sheet grid image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string", "description": "Root path of the dataset"},
                    "output_path": {
                        "type": "string",
                        "description": "Destination file path for the QA grid image",
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional specific image paths to inspect",
                    },
                },
                "required": ["dataset_dir", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_stratified_split",
            "description": (
                "Generate iterative multi-label stratified train/val/test image manifests "
                "preserving rare species proportions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string", "description": "Root path of the dataset"},
                    "output_dir": {"type": "string", "description": "Directory for split manifests"},
                },
                "required": ["dataset_dir", "output_dir"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are AgriData-Agent, an autonomous Computer Vision Data Engineering assistant
specialized in multi-class agricultural object detection datasets.
Follow a strict engineering workflow:
1. Always call `audit_dataset` first to inspect formats, pairing errors, class skew, and box anomalies.
2. Call `render_qa_contact_sheet` to generate visual evidence of flagged anomalous or high-density frames.
3. Call `create_stratified_split` to produce leak-free, rare-class-balanced train/val/test manifests.
4. Conclude with a concise, well-structured Markdown Engineering Audit Report summarizing findings,
class imbalance risks, visual QA outputs, and recommended next steps."""


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    handlers: Dict[str, Callable[..., Any]] = {
        "audit_dataset": lambda **kw: audit_dataset(**kw).model_dump_json(indent=2),
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
    """Executes the full autonomous multi-tool pipeline locally without requiring an external LLM API key."""
    audit = audit_dataset(dataset_dir)
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
- **Missing Annotations:** {len(audit.missing_annotations)}
- **Orphan Annotations:** {len(audit.orphan_annotations)}

## 2. Class Distribution & Imbalance Diagnostics
- **Max/Min Imbalance Ratio:** `{audit.imbalance_ratio}x`
- **Rare Classes (<15% of majority class):** {", ".join(audit.rare_classes) if audit.rare_classes else "None"}
- **Per-Class Instance Counts:** `{json.dumps(audit.class_counts)}`

## 3. Bounding-Box Geometry & Visual QA
- **Total Geometry/Format Anomalies Detected:** {audit.anomalies_found}
- **QA Contact Sheet Generated:** `{qa_path}`

## 4. Multi-Label Stratified Splits
- **Manifest Directory:** `{splits.output_dir}`
- **Train / Val / Test Counts:** `{splits.train_images}` / `{splits.val_images}` / `{splits.test_images}`
"""


def run_agent(
    dataset_dir: str,
    output_dir: str = "./splits",
    objective: str = "Audit dataset health, generate visual QA contact sheet, and build stratified splits.",
    max_steps: int = 8,
    offline: bool = False,
) -> str:
    """Runs the LLM tool-calling agent loop (or deterministic local pipeline if offline=True or no API key)."""
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
            "content": (
                f"Dataset directory: {dataset_dir}\n"
                f"Output directory: {output_dir}\n"
                f"Objective: {objective}"
            ),
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
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")
            tool_output = _dispatch_tool(fn_name, fn_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                }
            )

    return "Agent reached maximum reasoning steps before completing report."
