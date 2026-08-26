from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import reproduce_v2  # noqa: E402


class V2ReproductionTests(unittest.TestCase):
    def test_staging_restores_the_four_superseded_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged = reproduce_v2._stage_case(CASE_DIR, Path(temporary))
            for relative in reproduce_v2.V2_OVERLAYS:
                archived = CASE_DIR / "evidence" / "v2" / "code" / relative
                self.assertEqual(
                    hashlib.sha256((staged / relative).read_bytes()).hexdigest(),
                    hashlib.sha256(archived.read_bytes()).hexdigest(),
                )

    def test_archived_v2_replay_is_byte_identical(self) -> None:
        result = reproduce_v2.reproduce_v2()
        self.assertEqual(result["status"], "reproduced_byte_identical")
        self.assertEqual(result["exploratory_gate"], "fail")
        self.assertEqual(result["expected_replay_exit_code"], 1)
        self.assertFalse(result["model_rerun"])
        self.assertEqual(result["comparison"]["status"], "byte_identical")


if __name__ == "__main__":
    unittest.main()
