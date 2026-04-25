#!/usr/bin/env python3
"""Verify lightweight judge artifacts are parseable and render-safe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def check_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"path": str(path), "exists": True, "parseable": True, "type": type(payload).__name__}
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "parseable": False, "error": str(exc)}


def check_png(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        return {
            "path": str(path),
            "exists": True,
            "valid_png_signature": data.startswith(PNG_SIGNATURE),
            "bytes": len(data),
        }
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "valid_png_signature": False, "error": str(exc)}


def check_report(path: Path) -> dict[str, Any]:
    required = [
        "## Baseline Failure",
        "## Evidence Tracks",
        "## Compact RL Policy",
        "## LM-GRPO Attempt",
        "## Final Interpretation",
        "![Training reward curve]",
        "![Held-out comparison]",
    ]
    try:
        text = path.read_text(encoding="utf-8")
        missing = [item for item in required if item not in text]
        return {"path": str(path), "exists": True, "renders": not missing, "missing": missing}
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "renders": False, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check final artifact integrity.")
    parser.add_argument("--output", default="artifacts/eval/artifact_integrity_summary.json")
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = [
        Path("artifacts/eval/baseline_avg_reward.json"),
        Path("artifacts/eval/base_model_task3_eval.json"),
        Path("artifacts/eval/policy_gradient_task3_eval.json"),
        Path("artifacts/eval/lm_grpo_task3_eval_failed.json"),
        Path("artifacts/eval/before_after_trajectories.json"),
        Path("artifacts/eval/training_validation_summary.json"),
        Path("artifacts/eval/artifact_manifest.json"),
    ]
    png_paths = [
        Path("artifacts/plots/training_reward_curve.png"),
        Path("artifacts/plots/heldout_base_vs_trained.png"),
        Path("artifacts/plots/backup_training_reward_curve.png"),
    ]
    payload = {
        "json": [check_json(path) for path in json_paths],
        "png": [check_png(path) for path in png_paths],
        "report": check_report(Path("TRAINING_REPORT.md")),
    }
    payload["passed"] = (
        all(item.get("parseable") for item in payload["json"])
        and all(item.get("valid_png_signature") for item in payload["png"])
        and payload["report"].get("renders")
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"] and not args.allow_failed:
        raise SystemExit("Artifact integrity check failed.")


if __name__ == "__main__":
    main()
