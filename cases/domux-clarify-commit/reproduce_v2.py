#!/usr/bin/env python3
"""Reproduce the immutable exploratory v2 publication in a temporary tree.

The current policy contains post-v2 hardening.  This runner stages the exact
superseded v2 source files over a temporary copy, executes the frozen replay,
and requires all three generated artifacts to match the official record
byte-for-byte.  The recorded v2 gate intentionally fails, so replay exit code
1 is the expected successful reproduction signal before hash comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from verify_artifacts import (
    VerificationError,
    canonical_json,
    compare_replay_directory,
    verify_all,
)


CASE_DIR = Path(__file__).resolve().parent
V2_OVERLAYS = (
    "clarify_commit.py",
    "ha_acceptance.py",
    "tests/test_clarify_commit.py",
    "tests/test_ha_acceptance.py",
)


class ReproductionError(RuntimeError):
    """The frozen replay could not be reproduced exactly."""


def _stage_case(case_dir: Path, destination: Path) -> Path:
    """Copy the case and restore every superseded v2 source byte."""

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
    archive = case_dir / "evidence" / "v2" / "code"
    for relative in V2_OVERLAYS:
        source = archive / relative
        target = staged / relative
        if not source.is_file():
            raise ReproductionError(f"missing archived v2 source: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return staged


def reproduce_v2(
    case_dir: Path = CASE_DIR,
    *,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Run the archived replay once in a temporary tree and compare outputs."""

    case_dir = Path(case_dir).resolve()
    verify_all(case_dir)
    with tempfile.TemporaryDirectory(prefix="domux-v2-reproduction-") as temporary:
        root = Path(temporary)
        staged = _stage_case(case_dir, root)
        output = root / "output"
        completed = subprocess.run(
            [
                python_executable,
                str(staged / "replay_policy.py"),
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
                f"frozen v2 replay returned {completed.returncode}, expected 1"
                + (f": {detail}" if detail else "")
            )
        try:
            replay_status = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ReproductionError("frozen v2 replay emitted invalid JSON") from exc
        if replay_status.get("status") != "complete":
            raise ReproductionError("frozen v2 replay did not complete")
        if replay_status.get("exploratory_gate") != "fail":
            raise ReproductionError("frozen v2 replay gate no longer matches the record")
        comparison = compare_replay_directory(
            output,
            official_dir=case_dir / "evidence" / "v2",
        )
    return {
        "status": "reproduced_byte_identical",
        "analysis": "post-formal exploratory remediation replay",
        "exploratory_gate": "fail",
        "expected_replay_exit_code": 1,
        "model_rerun": False,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = reproduce_v2()
    except (
        OSError,
        ReproductionError,
        subprocess.SubprocessError,
        VerificationError,
    ) as exc:
        print(f"v2 reproduction failed: {exc}", file=sys.stderr)
        return 1
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
