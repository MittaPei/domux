from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import reproduce_v1  # noqa: E402


class V1ReproductionTests(unittest.TestCase):
    def test_staging_restores_all_archived_v1_sources(self) -> None:
        manifest = json.loads(
            (CASE_DIR / "evidence" / "v1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            staged = reproduce_v1._stage_case(CASE_DIR, Path(temporary))
            for relative in reproduce_v1.V1_OVERLAYS:
                self.assertEqual(
                    hashlib.sha256((staged / relative).read_bytes()).hexdigest(),
                    manifest["code"][relative],
                )

    def test_archived_v1_evaluation_is_byte_identical(self) -> None:
        result = reproduce_v1.reproduce_v1()
        self.assertEqual(result["status"], "reproduced_byte_identical")
        self.assertEqual(result["quality_gate"], "fail")
        self.assertEqual(result["expected_evaluator_exit_code"], 1)
        self.assertFalse(result["model_rerun"])
        self.assertEqual(result["comparison"]["status"], "byte_identical")


if __name__ == "__main__":
    unittest.main()
