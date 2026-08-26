from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import verify_artifacts  # noqa: E402
from verify_artifacts import VerificationError  # noqa: E402


def aggregate_inputs(
    version: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    report = json.loads(
        (CASE_DIR / "evidence" / version / "report.json").read_text(encoding="utf-8")
    )
    trials = [
        json.loads(line)
        for line in (CASE_DIR / "evidence" / version / "trials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    evaluation = [
        row
        for line in (CASE_DIR / "data" / "scenarios.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if (row := json.loads(line)).get("split") == "eval"
    ]
    return report, trials, evaluation


class ArtifactVerifierTests(unittest.TestCase):
    def test_default_verification_covers_all_evidence_sets(self) -> None:
        result = verify_artifacts.verify_all()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["v1"]["raw_probes"], 96)
        self.assertEqual(result["v1"]["trial_records"], 936)
        self.assertTrue(result["v1"]["formal_headline"])
        self.assertEqual(result["post_formal_diagnostic"]["base_count"], 48)
        self.assertEqual(
            result["post_formal_diagnostic"]["exact_delta_successes"], 16
        )
        self.assertEqual(
            result["post_formal_diagnostic"]["wrong_target_transitions"], 13
        )
        self.assertEqual(result["post_formal_diagnostic"]["model_calls"], 0)
        self.assertEqual(result["v2"]["trial_records"], 936)
        self.assertEqual(result["v2"]["exploratory_gate"], "fail")
        self.assertFalse(result["v2"]["model_rerun"])
        self.assertEqual(result["v3"]["policy_tests"], 106)
        self.assertEqual(result["v3"]["full_tests"], 184)
        self.assertEqual(result["v3"]["frozen_reproductions"], 2)
        self.assertFalse(result["v3"]["model_rerun"])
        self.assertFalse(result["v3"]["official_v2_replay"])
        self.assertEqual(result["v4"]["policy_tests"], 108)
        self.assertEqual(result["v4"]["ha_tests"], 13)
        self.assertEqual(result["v4"]["full_tests"], 188)
        self.assertEqual(result["v4"]["clean_room_tests"], 188)
        self.assertFalse(result["v4"]["model_rerun"])
        self.assertFalse(result["v4"]["official_v2_replay"])
        self.assertEqual(result["home_assistant"]["sut_cases"], 4)
        self.assertEqual(result["home_assistant"]["successful_transitions"], 3)
        self.assertEqual(result["home_assistant"]["rejected_before_dispatch"], 1)
        self.assertEqual(result["home_assistant"]["sut_dispatch_total"], 3)
        self.assertEqual(result["home_assistant"]["drift_sut_dispatch_delta"], 0)
        self.assertEqual(result["home_assistant"]["domux_evidence_pairs"], 4)
        self.assertEqual(result["home_assistant"]["total_service_calls"], 9)

    def test_cli_default_success_is_machine_readable(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = verify_artifacts.main([])
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue())["status"], "verified")

    def test_compare_accepts_an_exact_fresh_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for name in verify_artifacts.REPLAY_OUTPUT_NAMES:
                shutil.copyfile(CASE_DIR / "evidence" / "v2" / name, destination / name)
            result = verify_artifacts.compare_replay_directory(destination)
        self.assertEqual(result["status"], "byte_identical")
        self.assertEqual(set(result["files"]), set(verify_artifacts.REPLAY_OUTPUT_NAMES))

    def test_compare_rejects_a_single_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for name in verify_artifacts.REPLAY_OUTPUT_NAMES:
                shutil.copyfile(CASE_DIR / "evidence" / "v2" / name, destination / name)
            report = destination / "report.json"
            report.write_bytes(report.read_bytes() + b"\n")
            with self.assertRaisesRegex(VerificationError, "report.json"):
                verify_artifacts.compare_replay_directory(destination)

    def test_compare_rejects_a_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            shutil.copyfile(
                CASE_DIR / "evidence" / "v2" / "manifest.json",
                destination / "manifest.json",
            )
            with self.assertRaisesRegex(VerificationError, "comparison v2 trials.jsonl"):
                verify_artifacts.compare_replay_directory(destination)

    def test_cli_compare_failure_is_nonzero_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = verify_artifacts.main(["--compare", temporary])
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("verification failed:", stderr.getvalue())
        self.assertIn("comparison v2 trials.jsonl", stderr.getvalue())

    def test_aggregate_verification_detects_report_metric_tampering(self) -> None:
        report, trials, evaluation = aggregate_inputs("v2")
        report["metrics"]["language"]["candidate_coverage"]["successes"] = 47
        with self.assertRaisesRegex(VerificationError, "candidate_coverage successes"):
            verify_artifacts._verify_aggregates(report, trials, evaluation)

    def test_aggregate_verification_detects_duplicate_trials(self) -> None:
        report, trials, evaluation = aggregate_inputs("v2")
        with self.assertRaisesRegex(VerificationError, "duplicate language trial"):
            verify_artifacts._verify_aggregates(
                report,
                [*trials, copy.deepcopy(trials[0])],
                evaluation,
            )

    def test_aggregate_verification_recomputes_declared_statistics(self) -> None:
        report, trials, evaluation = aggregate_inputs("v1")
        mutations = (
            (
                "Wilson interval",
                lambda value: value["metrics"]["language"]["sensitivity"][
                    "wilson_95"
                ].update({"lower": 0.5}),
                "sensitivity Wilson lower bound mismatch",
            ),
            (
                "latency median",
                lambda value: value["metrics"]["language"]["latency"].update(
                    {"median_ms": 1.0}
                ),
                "language latency median mismatch",
            ),
            (
                "exact McNemar p",
                lambda value: value["primary_inference"]["comparisons"][0].update(
                    {"exact_two_sided_p": 0.5}
                ),
                "exact_two_sided_p mismatch",
            ),
            (
                "Holm adjustment",
                lambda value: value["primary_inference"]["comparisons"][1].update(
                    {"holm_adjusted_p": 0.5}
                ),
                "holm_adjusted_p mismatch",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(report)
                mutate(tampered)
                with self.assertRaisesRegex(VerificationError, expected_error):
                    verify_artifacts._verify_aggregates(
                        tampered,
                        trials,
                        evaluation,
                    )

    def test_aggregate_verification_requires_only_v1_primary_inference(self) -> None:
        v1_report, v1_trials, evaluation = aggregate_inputs("v1")
        del v1_report["primary_inference"]
        with self.assertRaisesRegex(
            VerificationError,
            "v1 primary inference is missing",
        ):
            verify_artifacts._verify_aggregates(
                v1_report,
                v1_trials,
                evaluation,
                primary_inference_required=True,
            )

        v2_report, v2_trials, _ = aggregate_inputs("v2")
        v2_report["primary_inference"] = {}
        with self.assertRaisesRegex(
            VerificationError,
            "v2 contains a primary inference section",
        ):
            verify_artifacts._verify_aggregates(
                v2_report,
                v2_trials,
                evaluation,
                primary_inference_required=False,
            )

    def test_metric_counts_reject_bool_values(self) -> None:
        report, trials, evaluation = aggregate_inputs("v1")
        report["metrics"]["execution"]["B0_unique_or_abstain"][
            "dispatch_coverage"
        ]["successes"] = True
        with self.assertRaisesRegex(
            VerificationError,
            "dispatch_coverage successes is not an integer",
        ):
            verify_artifacts._verify_aggregates(report, trials, evaluation)

    def test_latency_validates_present_value_when_pair_mate_is_missing(self) -> None:
        report, trials, evaluation = aggregate_inputs("v1")
        language = [trial for trial in trials if trial["record_type"] == "language_probe"]
        base_id = language[0]["base_id"]
        pair = [trial for trial in language if trial["base_id"] == base_id]
        next(trial for trial in pair if trial["variant"] == "clear")[
            "latency_ms"
        ] = None
        next(trial for trial in pair if trial["variant"] == "ambiguous")[
            "latency_ms"
        ] = True
        with self.assertRaisesRegex(
            VerificationError,
            f"language latency {base_id}/ambiguous is not a finite number",
        ):
            verify_artifacts._verify_aggregates(report, trials, evaluation)

    def test_holm_adjustment_handles_unequal_p_values(self) -> None:
        adjusted = verify_artifacts._with_holm_adjustment(
            [
                {"comparison_id": "higher", "exact_two_sided_p": 0.04},
                {"comparison_id": "lower", "exact_two_sided_p": 0.01},
            ]
        )
        self.assertEqual([item["comparison_id"] for item in adjusted], ["higher", "lower"])
        self.assertAlmostEqual(adjusted[0]["holm_adjusted_p"], 0.04)
        self.assertAlmostEqual(adjusted[1]["holm_adjusted_p"], 0.02)
        self.assertTrue(adjusted[0]["reject_at_0_05"])
        self.assertTrue(adjusted[1]["reject_at_0_05"])

    def test_pinned_first_match_diagnostic_rejects_byte_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            evidence = copied / "evidence" / "diagnostics" / "v1_first_match.json"
            evidence.write_bytes(evidence.read_bytes() + b"\n")
            _, evaluation = verify_artifacts._verify_frozen_data(copied)
            with self.assertRaisesRegex(
                VerificationError,
                "v1 first-match diagnostic hash mismatch",
            ):
                verify_artifacts._verify_v1_first_match_diagnostic(
                    copied,
                    evaluation,
                )

    def test_first_match_diagnostic_binds_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            source = copied / "diagnose_first_match_v1.py"
            source.write_bytes(source.read_bytes() + b"\n")
            _, evaluation = verify_artifacts._verify_frozen_data(copied)
            with self.assertRaisesRegex(
                VerificationError,
                "diagnostic diagnostic_code input hash mismatch",
            ):
                verify_artifacts._verify_v1_first_match_diagnostic(
                    copied,
                    evaluation,
                )

    def test_first_match_diagnostic_recomputes_semantic_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            evidence = copied / "evidence" / "diagnostics" / "v1_first_match.json"
            artifact = json.loads(evidence.read_text(encoding="utf-8"))
            artifact["comparison"]["post_formal_first_match"][
                "exact_delta_success"
            ]["successes"] = 17
            payload = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            evidence.write_bytes(payload)
            _, evaluation = verify_artifacts._verify_frozen_data(copied)
            with mock.patch.object(
                verify_artifacts,
                "PINNED_V1_FIRST_MATCH_DIAGNOSTIC_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ), self.assertRaisesRegex(
                VerificationError,
                "diagnostic exact_delta_success successes mismatch",
            ):
                verify_artifacts._verify_v1_first_match_diagnostic(
                    copied,
                    evaluation,
                )

    def test_first_match_diagnostic_rebuild_rejects_forged_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            evidence = copied / "evidence" / "diagnostics" / "v1_first_match.json"
            artifact = json.loads(evidence.read_text(encoding="utf-8"))
            instruction = next(
                instruction
                for trial in artifact["trials"]
                for instruction in trial["instructions"]
                if len(instruction["candidate_ids_in_frozen_inventory_order"]) > 1
            )
            instruction["candidate_ids_in_frozen_inventory_order"][1] = (
                "light.forged_candidate"
            )
            payload = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            evidence.write_bytes(payload)
            _, evaluation = verify_artifacts._verify_frozen_data(copied)
            with mock.patch.object(
                verify_artifacts,
                "PINNED_V1_FIRST_MATCH_DIAGNOSTIC_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ), self.assertRaisesRegex(
                VerificationError,
                "differs from its deterministic rebuild",
            ):
                verify_artifacts._verify_v1_first_match_diagnostic(
                    copied,
                    evaluation,
                )

    def test_pinned_v2_manifest_rejects_a_tampered_case_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(CASE_DIR, copied, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
            report = copied / "evidence" / "v2" / "manifest.json"
            report.write_bytes(report.read_bytes() + b"\n")
            with self.assertRaisesRegex(VerificationError, "v2 publication manifest hash mismatch"):
                verify_artifacts.verify_all(copied)

    def test_v1_archived_source_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(CASE_DIR, copied, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
            source = copied / "evidence" / "v1" / "code" / "clarify_commit.py"
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(VerificationError, "archived v1 clarify_commit.py hash mismatch"):
                verify_artifacts.verify_all(copied)

    def test_superseded_v2_source_archive_is_the_freeze_authority(self) -> None:
        for relative in verify_artifacts.V2_ARCHIVED_SOURCE_FILES:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "case"
                shutil.copytree(
                    CASE_DIR,
                    copied,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
                )
                source = copied / "evidence" / "v2" / "code" / relative
                source.write_bytes(source.read_bytes() + b"\n")
                with self.assertRaisesRegex(
                    VerificationError,
                    f"v2 source {relative} hash mismatch",
                ):
                    verify_artifacts.verify_all(copied)

    def test_pinned_ha_record_rejects_a_tampered_case_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(CASE_DIR, copied, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
            evidence = copied / "evidence" / "ha_acceptance.json"
            evidence.write_bytes(evidence.read_bytes() + b"\n")
            with self.assertRaisesRegex(VerificationError, "Home Assistant acceptance hash mismatch"):
                verify_artifacts.verify_all(copied)

        mutations = (
            (
                "readiness",
                lambda artifact: artifact["home_assistant"]["readiness"].update(
                    {"endpoint": "/api/"}
                ),
                "HA readiness changed",
            ),
            (
                "onboarding",
                lambda artifact: artifact["home_assistant"]["onboarding"][
                    "initial"
                ].update({"analytics": True}),
                "HA onboarding changed",
            ),
            (
                "aggregate",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"].update(
                    {"case_count": 5}
                ),
                "HA acceptance case count mismatch",
            ),
            (
                "binding",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"]["cases"][3][
                    "binding"
                ].update({
                    "after_external_mutation_state_digest": artifact["home_assistant"][
                        "phases"
                    ]["sut"]["cases"][3]["binding"]["prepared_state_digest"],
                }),
                "HA target-drift state binding changed",
            ),
            (
                "exact drift digest",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"][
                    "cases"
                ][3]["binding"].update({"prepared_state_digest": "0" * 64}),
                "HA target-drift state binding changed",
            ),
            (
                "rejection",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"]["cases"][3][
                    "rejection"
                ].update({"sut_dispatch_delta": 1}),
                "HA target-drift rejection changed",
            ),
            (
                "ledger",
                lambda artifact: artifact["home_assistant"]["phases"][
                    "service_call_accounting"
                ].update({"total": 7}),
                "HA service-call accounting changed",
            ),
            (
                "provenance line",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"][
                    "cases"
                ][0]["domux_evidence"].update({"line_number": 3}),
                "HA Domux provenance changed",
            ),
            (
                "provenance raw hash",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"][
                    "cases"
                ][1]["domux_evidence"].update({"raw_output_sha256": "0" * 64}),
                "HA Domux provenance changed",
            ),
            (
                "scenario target mapping",
                lambda artifact: artifact["home_assistant"]["phases"]["sut"][
                    "cases"
                ][0]["scenario_provenance"][
                    "scenario_target_to_ha_demo_entity"
                ].update({"ha_demo_entity_id": "light.bed_light"}),
                "HA scenario provenance changed",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(semantic_tamper=name), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "case"
                shutil.copytree(
                    CASE_DIR,
                    copied,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
                )
                evidence = copied / "evidence" / "ha_acceptance.json"
                artifact = json.loads(evidence.read_text(encoding="utf-8"))
                mutate(artifact)
                payload = (
                    json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
                evidence.write_bytes(payload)
                with mock.patch.object(
                    verify_artifacts,
                    "PINNED_HA_ACCEPTANCE_SHA256",
                    hashlib.sha256(payload).hexdigest(),
                ), self.assertRaisesRegex(VerificationError, expected_error):
                    verify_artifacts._verify_ha(copied)

    def test_ha_provenance_rejects_tampered_v1_pairs_and_duplicate_keys(self) -> None:
        mutations = (
            (
                "query hash",
                lambda records: records[1].update({"query_sha256": "0" * 64}),
                "v1 Domux query hash mismatch",
            ),
            (
                "duplicate key",
                lambda records: records[3].update(
                    {
                        "base_id": records[2]["base_id"],
                        "variant": records[2]["variant"],
                    }
                ),
                "v1 Domux raw evidence contains duplicate key",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "case"
                shutil.copytree(
                    CASE_DIR,
                    copied,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
                )
                evidence = copied / "evidence" / "v1" / "domux_raw.jsonl"
                records = [
                    json.loads(line)
                    for line in evidence.read_text(encoding="utf-8").splitlines()
                ]
                mutate(records)
                payload = (
                    "".join(
                        verify_artifacts.canonical_json(record) + "\n"
                        for record in records
                    )
                ).encode("utf-8")
                evidence.write_bytes(payload)
                raw_sha256 = hashlib.sha256(payload).hexdigest()
                ha_path = copied / "evidence" / "ha_acceptance.json"
                ha = json.loads(ha_path.read_text(encoding="utf-8"))
                sut = ha["home_assistant"]["phases"]["sut"]
                sut["domux_evidence"]["artifact_sha256"] = raw_sha256
                for case in sut["cases"]:
                    case["domux_evidence"]["artifact_sha256"] = raw_sha256
                ha_payload = (
                    json.dumps(ha, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                ha_path.write_bytes(ha_payload)
                with mock.patch.object(
                    verify_artifacts,
                    "PINNED_V1_DOMUX_RAW_SHA256",
                    raw_sha256,
                ), mock.patch.object(
                    verify_artifacts,
                    "PINNED_HA_ACCEPTANCE_SHA256",
                    hashlib.sha256(ha_payload).hexdigest(),
                ), self.assertRaisesRegex(VerificationError, expected_error):
                    verify_artifacts._verify_ha(copied)

    def test_v4_uses_its_archived_home_assistant_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            current = copied / "evidence" / "ha_acceptance.json"
            current.write_bytes(current.read_bytes() + b"\n")
            self.assertEqual(verify_artifacts._verify_v4(copied)["status"], "verified")

            archived = copied / "evidence" / "v4" / "ha_acceptance.json"
            archived.write_bytes(archived.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                VerificationError,
                "Home Assistant acceptance hash mismatch",
            ):
                verify_artifacts._verify_v4(copied)

    def test_pinned_v3_validation_rejects_a_tampered_case_copy(self) -> None:
        for version in ("v3", "v4"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "case"
                shutil.copytree(
                    CASE_DIR,
                    copied,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
                )
                validation = copied / "evidence" / version / "validation.json"
                validation.write_bytes(validation.read_bytes() + b"\n")
                with self.assertRaisesRegex(
                    VerificationError,
                    f"{version} validation hash mismatch",
                ):
                    verify_artifacts.verify_all(copied)

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            validation = copied / "evidence" / "v4" / "validation.json"
            artifact = json.loads(validation.read_text(encoding="utf-8"))
            artifact["validation_results"]["case_full_suite"]["passed"] = 189
            payload = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            validation.write_bytes(payload)
            with mock.patch.object(
                verify_artifacts,
                "PINNED_V4_VALIDATION_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ), self.assertRaisesRegex(
                VerificationError,
                "v4 full-suite result changed",
            ):
                verify_artifacts._verify_v4(copied)

    def test_v3_source_binding_rejects_archived_policy_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            policy = copied / "evidence" / "v3" / "code" / "clarify_commit.py"
            policy.write_bytes(policy.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                VerificationError,
                "v3 source clarify_commit.py hash mismatch",
            ):
                verify_artifacts.verify_all(copied)

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            for relative in verify_artifacts.V4_ARCHIVED_SOURCE_PATHS:
                source = copied / relative
                source.write_bytes(source.read_bytes() + b"\n")
            self.assertEqual(verify_artifacts._verify_v4(copied)["status"], "verified")

        for relative in verify_artifacts.V4_SOURCE_FILES:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "case"
                shutil.copytree(
                    CASE_DIR,
                    copied,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
                )
                source = copied / verify_artifacts.V4_ARCHIVED_SOURCE_PATHS.get(
                    relative,
                    relative,
                )
                source.write_bytes(source.read_bytes() + b"\n")
                with self.assertRaisesRegex(
                    VerificationError,
                    f"v4 source {relative} hash mismatch",
                ):
                    verify_artifacts._verify_v4(copied)


if __name__ == "__main__":
    unittest.main()
