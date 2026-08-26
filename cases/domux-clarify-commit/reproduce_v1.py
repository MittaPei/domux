#!/usr/bin/env python3
"""Reproduce the immutable formal v1 report from frozen model outputs.

The runner stages the archived v1 policy and evaluator at the case root, where
their registered data paths are valid, then requires the regenerated report
and trials to match the official artifacts byte-for-byte.  The formal quality
gate intentionally failed, so evaluator exit code 1 is expected before the
hash comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from verify_artifacts import VerificationError, canonical_json, verify_all


CASE_DIR = Path(__file__).resolve().parent
V1_OVERLAYS = (
    "clarify_commit.py",
    "evaluate.py",
    "run_model.py",
)
V1_OUTPUT_NAMES = ("trials.jsonl", "report.json")


class ReproductionError(RuntimeError):
    """The frozen formal evaluation could not be reproduced exactly."""


def _stage_case(case_dir: Path, destination: Path) -> Path:
    case_dir = Path(case_dir).resolve()
    for path in case_dir.rglob("*"):
        if path.is_symlink():
            raise ReproductionError(f"refusing to stage symbolic link: {path}")
    staged = destination / "case"
    shutil.copytree(
        case_dir,
        staged,
        ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache", "*.pyc"),
    )
    archive = case_dir / "evidence" / "v1" / "code"
    for relative in V1_OVERLAYS:
        source = archive / relative
        target = staged / relative
        if not source.is_file():
            raise ReproductionError(f"missing archived v1 source: {relative}")
        shutil.copyfile(source, target)
    return staged


def _compare_v1(output: Path, official: Path) -> dict[str, object]:
    digests: dict[str, str] = {}
    for name in V1_OUTPUT_NAMES:
        candidate = (output / name).read_bytes()
        expected = (official / name).read_bytes()
        if candidate != expected:
            raise ReproductionError(f"reproduced v1 artifact differs: {name}")
        digests[name] = hashlib.sha256(expected).hexdigest()
    return {"status": "byte_identical", "files": digests}


def reproduce_v1(
    case_dir: Path = CASE_DIR,
    *,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    case_dir = Path(case_dir).resolve()
    verify_all(case_dir)
    with tempfile.TemporaryDirectory(prefix="domux-v1-reproduction-") as temporary:
        root = Path(temporary)
        staged = _stage_case(case_dir, root)
        output = root / "output"
        evidence = staged / "evidence" / "v1"
        completed = subprocess.run(
            [
                python_executable,
                str(staged / "evaluate.py"),
                "--from-frozen-evidence",
                str(evidence / "domux_raw.jsonl"),
                "--evidence-metadata",
                str(evidence / "model_metadata.json"),
                "--output-dir",
                str(output),
            ],
            cwd=staged,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 1:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ReproductionError(
                f"frozen v1 evaluator returned {completed.returncode}, expected 1"
                + (f": {detail}" if detail else "")
            )
        try:
            status = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ReproductionError("frozen v1 evaluator emitted invalid JSON") from exc
        if status.get("status") != "complete" or status.get("quality_gate") != "fail":
            raise ReproductionError("frozen v1 result classification changed")
        comparison = _compare_v1(output, case_dir / "evidence" / "v1")
    return {
        "status": "reproduced_byte_identical",
        "analysis": "sole pre-remediation formal evaluation",
        "quality_gate": "fail",
        "expected_evaluator_exit_code": 1,
        "model_rerun": False,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = reproduce_v1()
    except (
        OSError,
        ReproductionError,
        subprocess.SubprocessError,
        VerificationError,
    ) as exc:
        print(f"v1 reproduction failed: {exc}", file=sys.stderr)
        return 1
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
