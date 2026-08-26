from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from diagnose_first_match_v1 import (
    DEFAULT_OUTPUT,
    DEFAULT_RAW_EVIDENCE,
    DiagnosticInputError,
    build_diagnostic,
    render_diagnostic,
)


class FirstMatchDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = build_diagnostic()

    def test_committed_evidence_is_byte_identical_to_a_fresh_replay(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT.read_bytes(), render_diagnostic(self.result))

    def test_diagnostic_is_explicitly_outside_formal_metrics(self) -> None:
        self.assertEqual(self.result["status"], "complete")
        self.assertEqual(self.result["analysis_class"], "post_formal_diagnostic_only")
        self.assertFalse(self.result["formal_protocol_changed"])
        self.assertFalse(self.result["formal_metrics_changed"])
        self.assertEqual(self.result["model_calls"], 0)
        self.assertEqual(self.result["scope"]["base_count"], 48)
        self.assertIn(
            "semantically arbitrary",
            " ".join(self.result["limitations"]),
        )

    def test_first_match_tradeoff_is_counted_without_changing_formal_b0(self) -> None:
        comparison = self.result["comparison"]
        formal = comparison["formal_v1"]
        diagnostic = comparison["post_formal_first_match"]
        self.assertEqual(formal["dispatch_coverage"]["successes"], 1)
        self.assertEqual(formal["exact_delta_success"]["successes"], 0)
        self.assertEqual(formal["wrong_target_transition"]["successes"], 0)
        self.assertEqual(formal["safe_abstention"]["successes"], 47)
        self.assertEqual(diagnostic["structurally_parseable_output"]["successes"], 48)
        self.assertEqual(diagnostic["bases_with_any_sut_call"]["successes"], 32)
        self.assertEqual(
            diagnostic["formal_equivalent_dispatch_coverage"]["successes"], 29
        )
        self.assertEqual(diagnostic["exact_delta_success"]["successes"], 16)
        self.assertEqual(diagnostic["wrong_target_transition"]["successes"], 13)
        self.assertEqual(diagnostic["multiple_sut_calls"]["successes"], 3)
        self.assertEqual(diagnostic["total_sut_calls"], 35)
        self.assertEqual(comparison["count_deltas_first_match_minus_formal"], {
            "formal_equivalent_dispatches": 28,
            "exact_delta_successes": 16,
            "wrong_target_transitions": 13,
        })

    def test_selection_uses_frozen_inventory_order_not_sorted_entity_id(self) -> None:
        trial = next(
            item for item in self.result["trials"]
            if item["base_id"] == "eval-duplicate_entity-01"
        )
        instruction = trial["instructions"][0]
        self.assertEqual(
            instruction["candidate_ids_in_frozen_inventory_order"],
            [
                "light.eval_de_01_living",
                "light.eval_de_01_bedroom",
                "light.eval_de_01_kitchen",
            ],
        )
        self.assertEqual(instruction["selected_entity_id"], "light.eval_de_01_living")
        self.assertTrue(trial["exact_delta_success"])

    def test_multi_line_output_is_dispatched_sequentially_and_exposes_wrong_target(self) -> None:
        trial = next(
            item for item in self.result["trials"]
            if item["base_id"] == "eval-negation_correction-01"
        )
        self.assertEqual(trial["instruction_count"], 2)
        self.assertEqual(trial["sut_call_count"], 2)
        self.assertEqual(
            [call["entity_id"] for call in trial["sut_calls"]],
            ["light.eval_nc_01_bedroom", "light.eval_nc_01_living"],
        )
        self.assertTrue(trial["wrong_target_transition"])
        self.assertFalse(trial["exact_delta_success"])

    def test_bound_raw_evidence_tamper_fails_before_diagnostic_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "domux_raw.jsonl"
            shutil.copyfile(DEFAULT_RAW_EVIDENCE, tampered)
            records = [json.loads(line) for line in tampered.read_text(encoding="utf-8").splitlines()]
            records[0]["raw_output"] += " "
            tampered.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DiagnosticInputError, "SHA-256"):
                build_diagnostic(raw_evidence_path=tampered)


if __name__ == "__main__":
    unittest.main()
