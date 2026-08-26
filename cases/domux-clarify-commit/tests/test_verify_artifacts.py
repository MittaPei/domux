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


class ArtifactVerifierTests(unittest.TestCase):
    def test_default_verification_covers_all_evidence_sets(self) -> None:
        result = verify_artifacts.verify_all()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["v1"]["raw_probes"], 96)
        self.assertEqual(result["v1"]["trial_records"], 936)
        self.assertTrue(result["v1"]["formal_headline"])
        self.assertEqual(result["v2"]["trial_records"], 936)
        self.assertEqual(result["v2"]["exploratory_gate"], "fail")
        self.assertFalse(result["v2"]["model_rerun"])
        self.assertEqual(result["v3"]["policy_tests"], 106)
        self.assertEqual(result["v3"]["full_tests"], 184)
        self.assertEqual(result["v3"]["frozen_reproductions"], 2)
        self.assertFalse(result["v3"]["model_rerun"])
        self.assertFalse(result["v3"]["official_v2_replay"])
        self.assertEqual(result["home_assistant"]["sut_cases"], 4)
        self.assertEqual(result["home_assistant"]["successful_transitions"], 3)
        self.assertEqual(result["home_assistant"]["rejected_before_dispatch"], 1)
        self.assertEqual(result["home_assistant"]["sut_dispatch_total"], 3)
        self.assertEqual(result["home_assistant"]["drift_sut_dispatch_delta"], 0)

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
        report = json.loads(
            (CASE_DIR / "evidence" / "v2" / "report.json").read_text(encoding="utf-8")
        )
        trials = [
            json.loads(line)
            for line in (CASE_DIR / "evidence" / "v2" / "trials.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        evaluation = [
            json.loads(line)
            for line in (CASE_DIR / "data" / "scenarios.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line).get("split") == "eval"
        ]
        report["metrics"]["language"]["candidate_coverage"]["successes"] = 47
        with self.assertRaisesRegex(VerificationError, "candidate_coverage successes"):
            verify_artifacts._verify_aggregates(report, trials, evaluation)

    def test_aggregate_verification_detects_duplicate_trials(self) -> None:
        report = json.loads(
            (CASE_DIR / "evidence" / "v2" / "report.json").read_text(encoding="utf-8")
        )
        trials = [
            json.loads(line)
            for line in (CASE_DIR / "evidence" / "v2" / "trials.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        evaluation = [
            json.loads(line)
            for line in (CASE_DIR / "data" / "scenarios.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line).get("split") == "eval"
        ]
        with self.assertRaisesRegex(VerificationError, "duplicate language trial"):
            verify_artifacts._verify_aggregates(
                report,
                [*trials, copy.deepcopy(trials[0])],
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

    def test_pinned_v3_validation_rejects_a_tampered_case_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(
                CASE_DIR,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"),
            )
            validation = copied / "evidence" / "v3" / "validation.json"
            validation.write_bytes(validation.read_bytes() + b"\n")
            with self.assertRaisesRegex(VerificationError, "v3 validation hash mismatch"):
                verify_artifacts.verify_all(copied)

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


if __name__ == "__main__":
    unittest.main()
