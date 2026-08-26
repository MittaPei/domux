from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import replay_policy  # noqa: E402
from replay_policy import ReplayInputError, replay_to_directory  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_evaluation(
    _eval_rows: object,
    _protocol: object,
    _evidence: object,
    *,
    integrity: dict[str, object],
    model_run: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Exercise publication plumbing without running or inspecting v2 scores."""

    trials = [{"synthetic_publication_test_record": index} for index in range(936)]
    report = {
        "schema_version": 1,
        "status": "complete",
        "population": {"evaluation_bases": 48, "paired_probes": 96},
        "trial_counts": {
            "language_probe_records": 96,
            "execution_trial_records": 840,
            "total_records": 936,
        },
        "input_integrity": integrity,
        "model_run": model_run,
        "methods": {
            "binary_intervals": "synthetic publication test",
            "trial_reset": "synthetic publication test",
            "pseudo_replication_guards": "synthetic publication test",
        },
        "primary_inference": {"not_used": True},
        "quality_gate": {"result": "pass"},
        "determinism": {},
    }
    return trials, report


def write_code_freeze(path: Path, *, source_commit: str = "b" * 40) -> None:
    files: dict[str, dict[str, object]] = {}
    for relative in replay_policy.CODE_FREEZE_FILES:
        payload = (CASE_DIR / relative).read_bytes()
        files[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    canonical_bundle = json.dumps(
        {"files": files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "manifest_type": "domux-v2-code-freeze",
        "evidence_version": "v2-post-formal-exploratory",
        "status": "frozen-before-official-replay",
        "hash_algorithm": "sha256",
        "authority": "content-addressed-source-bundle",
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_bundle).hexdigest(),
        "fork_git_provenance": {
            "source_commit": source_commit,
            "role": "informational-only; content hashes are authoritative",
        },
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def publication_payloads() -> dict[str, bytes]:
    trials = b'{"trial":1}\n'
    report = b'{"status":"complete"}\n'
    outputs = {
        "trials.jsonl": {
            "sha256": hashlib.sha256(trials).hexdigest(),
            "size_bytes": len(trials),
        },
        "report.json": {
            "sha256": hashlib.sha256(report).hexdigest(),
            "size_bytes": len(report),
        },
    }
    manifest = (
        json.dumps({"outputs": outputs}, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return {"trials.jsonl": trials, "report.json": report, "manifest.json": manifest}


class ReplayPolicyTests(unittest.TestCase):
    """Temporary reproducibility checks only; these are not v2 evidence runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.first = cls.root / "reproducibility-check-a"
        cls.second = cls.root / "reproducibility-check-b"
        cls.code_freeze = cls.root / "code-freeze.json"
        write_code_freeze(cls.code_freeze)
        cls.patches = (
            mock.patch.object(
                replay_policy,
                "_execute_evaluation",
                side_effect=synthetic_evaluation,
            ),
        )
        for patcher in cls.patches:
            patcher.start()
        cls.report = cls._replay(output_dir=cls.first)
        cls._replay(output_dir=cls.second)

    @classmethod
    def _replay(cls, **kwargs: object) -> dict[str, object]:
        kwargs.setdefault("code_freeze_path", cls.code_freeze)
        return replay_to_directory(**kwargs)

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in reversed(cls.patches):
            patcher.stop()
        cls.temporary.cleanup()

    def test_replays_every_probe_and_trial_with_fixed_denominators(self) -> None:
        report = self.report
        self.assertEqual(report["population"]["evaluation_bases"], 48)
        self.assertEqual(report["population"]["paired_probes"], 96)
        self.assertEqual(report["input_integrity"]["evidence_pairs_verified"], 96)
        self.assertEqual(report["trial_counts"]["language_probe_records"], 96)
        self.assertEqual(report["trial_counts"]["execution_trial_records"], 840)
        self.assertEqual(report["trial_counts"]["total_records"], 936)
        trial_lines = (self.first / "trials.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(trial_lines), 936)
        self.assertTrue(all(isinstance(json.loads(line), dict) for line in trial_lines))

    def test_report_is_explicitly_exploratory_and_keeps_v1_formal(self) -> None:
        report = self.report
        classification = report["analysis_classification"]
        self.assertEqual(
            classification["stage"],
            "post-formal exploratory remediation replay",
        )
        self.assertFalse(classification["held_out"])
        self.assertFalse(classification["pre_registered"])
        self.assertFalse(classification["confirmatory"])
        self.assertTrue(classification["v1_remains_sole_formal"])
        self.assertEqual(classification["formal_headline_version"], "v1-formal")
        self.assertFalse(classification["model_rerun"])
        self.assertEqual(classification["reused_v1_raw_outputs"], 96)
        self.assertEqual(classification["recorded_evidence_publication_count"], 1)
        self.assertEqual(
            classification["publication_count_scope"],
            "one content-addressed official v2 record",
        )
        self.assertFalse(classification["byte_identical_reproduction_is_new_record"])
        self.assertNotIn("primary_inference", report)
        self.assertNotIn("quality_gate", report)
        self.assertEqual(report["methods"]["confirmatory_p_values"], "omitted")
        self.assertIn("descriptive", report["exploratory_gate"]["interpretation"])
        self.assertTrue(report["exploratory_gate"]["affects_process_exit"])

    def test_v1_and_current_code_bindings_are_visible_and_verified(self) -> None:
        binding = self.report["input_integrity"]["code_binding"]
        self.assertTrue(binding["v1_cross_bindings_verified"])
        self.assertTrue(binding["current_evaluator_matches_v1"])
        self.assertTrue(binding["current_runner_matches_v1"])
        self.assertTrue(binding["current_policy_differs_from_v1"])
        self.assertTrue(binding["v1_archived_source_bindings_verified"])
        source = binding["v2_code_freeze"]
        self.assertEqual(source["fork_source_commit"], "b" * 40)
        self.assertFalse(source["fork_commit_required_for_replay"])
        self.assertTrue(source["squash_merge_safe"])
        self.assertEqual(source["file_count"], len(replay_policy.CODE_FREEZE_FILES))
        self.assertNotEqual(
            binding["v1_grounding_policy_sha256"],
            binding["current_grounding_policy_sha256"],
        )
        self.assertEqual(
            binding["current_evaluator_sha256"],
            sha256_file(CASE_DIR / "evaluate.py"),
        )
        self.assertEqual(
            binding["current_replay_policy_sha256"],
            sha256_file(CASE_DIR / "replay_policy.py"),
        )
        self.assertTrue(self.report["model_run"]["metadata_verified"])
        self.assertEqual(self.report["model_run"]["artifact_origin"], "v1-formal")
        self.assertTrue(self.report["model_run"]["replayed_without_model_inference"])

    def test_output_bytes_are_deterministic_and_private(self) -> None:
        for name in replay_policy.OUTPUT_NAMES:
            first = self.first / name
            second = self.second / name
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(second.stat().st_mode), 0o600)

    def test_manifest_is_the_complete_hash_bound_publication_marker(self) -> None:
        manifest = json.loads((self.first / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            manifest["manifest_type"],
            "domux-v2-policy-replay-publication",
        )
        self.assertEqual(manifest["publication"]["completion_marker"], "manifest.json")
        self.assertTrue(manifest["publication"]["marker_written_last"])
        self.assertFalse(
            manifest["publication"]["byte_identical_reproduction_is_new_record"]
        )
        self.assertTrue(manifest["source_binding"]["squash_merge_safe"])
        self.assertFalse(manifest["code_binding"]["requires_fork_git_history"])
        for name in replay_policy.ARTIFACT_NAMES:
            payload = (self.first / name).read_bytes()
            self.assertEqual(manifest["outputs"][name]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(manifest["outputs"][name]["size_bytes"], len(payload))
        self.assertEqual(manifest["outputs"]["trials.jsonl"]["record_count"], 936)
        self.assertEqual(
            manifest["code_binding"]["provenance_only_code"].keys(),
            {"run_model.py"},
        )

    def test_existing_output_is_never_overwritten(self) -> None:
        before = {
            name: (self.first / name).read_bytes()
            for name in replay_policy.OUTPUT_NAMES
        }
        with self.assertRaisesRegex(ReplayInputError, "refusing to overwrite"):
            self._replay(output_dir=self.first)
        after = {
            name: (self.first / name).read_bytes()
            for name in replay_policy.OUTPUT_NAMES
        }
        self.assertEqual(before, after)

    def _copy_and_tamper(self, source: Path, name: str) -> Path:
        destination = self.root / name
        shutil.copy2(source, destination)
        destination.write_bytes(destination.read_bytes() + b"\n")
        return destination

    def test_tampered_raw_evidence_is_rejected_without_outputs(self) -> None:
        evidence = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_EVIDENCE,
            "tampered-raw.jsonl",
        )
        output = self.root / "tampered-raw-output"
        with self.assertRaisesRegex(ReplayInputError, "raw evidence hash"):
            self._replay(evidence_path=evidence, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_metadata_is_rejected_without_outputs(self) -> None:
        metadata = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_METADATA,
            "tampered-metadata.json",
        )
        output = self.root / "tampered-metadata-output"
        with self.assertRaisesRegex(ReplayInputError, "metadata hash"):
            self._replay(metadata_path=metadata, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_v1_report_is_rejected_without_outputs(self) -> None:
        report = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_REPORT,
            "tampered-v1-report.json",
        )
        output = self.root / "tampered-v1-report-output"
        with self.assertRaisesRegex(ReplayInputError, "report hash"):
            self._replay(v1_report_path=report, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_v1_trials_are_rejected_without_outputs(self) -> None:
        trials = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_TRIALS,
            "tampered-v1-trials.jsonl",
        )
        output = self.root / "tampered-v1-trials-output"
        with self.assertRaisesRegex(ReplayInputError, "trials hash"):
            self._replay(v1_trials_path=trials, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_manifest_is_rejected_without_outputs(self) -> None:
        manifest = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_MANIFEST,
            "tampered-manifest.json",
        )
        output = self.root / "tampered-manifest-output"
        with self.assertRaisesRegex(ReplayInputError, "manifest differs"):
            self._replay(manifest_path=manifest, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_policy_plan_is_rejected_without_outputs(self) -> None:
        plan = self._copy_and_tamper(
            replay_policy.DEFAULT_POLICY_PLAN,
            "tampered-plan.json",
        )
        output = self.root / "tampered-plan-output"
        with self.assertRaisesRegex(ReplayInputError, "policy plan differs"):
            self._replay(policy_plan_path=plan, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_protocol_is_rejected_without_outputs(self) -> None:
        protocol = self._copy_and_tamper(
            replay_policy.DEFAULT_PROTOCOL,
            "tampered-protocol.json",
        )
        output = self.root / "tampered-protocol-output"
        with self.assertRaisesRegex(ReplayInputError, "protocol differs"):
            self._replay(protocol_path=protocol, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_freeze_manifest_is_rejected_without_outputs(self) -> None:
        freeze = self._copy_and_tamper(
            replay_policy.DEFAULT_FREEZE,
            "tampered-freeze.json",
        )
        output = self.root / "tampered-freeze-output"
        with self.assertRaisesRegex(ReplayInputError, "freeze manifest differs"):
            self._replay(freeze_path=freeze, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_code_freeze_binding_is_rejected_without_outputs(self) -> None:
        manifest = json.loads(self.code_freeze.read_text(encoding="utf-8"))
        manifest["files"]["clarify_commit.py"]["sha256"] = "0" * 64
        tampered = self.root / "tampered-code-freeze.json"
        tampered.write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "tampered-code-freeze-output"
        with self.assertRaisesRegex(ReplayInputError, "source hash mismatch"):
            self._replay(code_freeze_path=tampered, output_dir=output)
        self.assertFalse(output.exists())

    def test_tampered_archived_v1_source_is_rejected_without_outputs(self) -> None:
        archive = self.root / "tampered-v1-code"
        shutil.copytree(replay_policy.DEFAULT_V1_CODE_DIR, archive)
        target = archive / "clarify_commit.py"
        target.write_bytes(target.read_bytes() + b"\n")
        output = self.root / "tampered-v1-code-output"
        with self.assertRaisesRegex(ReplayInputError, "archived v1 source hash mismatch"):
            self._replay(v1_code_dir=archive, output_dir=output)
        self.assertFalse(output.exists())

    def test_nonexistent_fork_commit_is_informational_only(self) -> None:
        source = self.report["input_integrity"]["code_binding"]["v2_code_freeze"]
        self.assertEqual(source["fork_source_commit"], "b" * 40)
        self.assertFalse(source["fork_commit_required_for_replay"])
        self.assertTrue(source["squash_merge_safe"])

    def test_manifest_last_publish_removes_partial_set_after_a_race(self) -> None:
        output = self.root / "raced-output"
        original_link = os.link
        calls = 0

        def raced_link(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FileExistsError(str(destination))
            original_link(source, destination)

        with mock.patch.object(replay_policy.os, "link", side_effect=raced_link):
            with self.assertRaisesRegex(ReplayInputError, "refusing to overwrite"):
                replay_policy._publish_outputs_with_completion_marker(
                    output,
                    publication_payloads(),
                )
        for name in replay_policy.OUTPUT_NAMES:
            self.assertFalse((output / name).exists())
        self.assertEqual(list(output.glob(".replay-policy-*")), [])

    def test_verification_failure_never_exposes_completion_marker(self) -> None:
        output = self.root / "verification-failure-output"
        original_read = replay_policy._read_bytes

        def fail_after_artifact_publish(path: Path, label: str) -> bytes:
            if label == "published report.json":
                raise ReplayInputError("injected read-back failure")
            return original_read(path, label)

        with mock.patch.object(
            replay_policy,
            "_read_bytes",
            side_effect=fail_after_artifact_publish,
        ), self.assertRaisesRegex(ReplayInputError, "injected"):
            replay_policy._publish_outputs_with_completion_marker(
                output,
                publication_payloads(),
            )
        for name in replay_policy.OUTPUT_NAMES:
            self.assertFalse((output / name).exists())
        self.assertEqual(list(output.glob(".replay-policy-*")), [])

    def test_cli_returns_nonzero_when_exploratory_gate_fails(self) -> None:
        fake_report = {
            "status": "complete",
            "analysis_classification": {"stage": "post-formal exploratory"},
            "exploratory_gate": {"result": "fail"},
            "population": {"evaluation_bases": 48},
            "trial_counts": {"total_records": 936},
        }
        with mock.patch.object(
            replay_policy,
            "replay_to_directory",
            return_value=fake_report,
        ), redirect_stdout(io.StringIO()):
            result = replay_policy.main(["--output-dir", str(self.root / "mock")])
        self.assertEqual(result, 1)

    def test_cli_returns_nonzero_for_tampered_input(self) -> None:
        evidence = self._copy_and_tamper(
            replay_policy.DEFAULT_V1_EVIDENCE,
            "cli-tampered-raw.jsonl",
        )
        output = self.root / "cli-tampered-output"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = replay_policy.main([
                "--v1-evidence",
                str(evidence),
                "--code-freeze",
                str(self.code_freeze),
                "--output-dir",
                str(output),
            ])
        self.assertNotEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "error")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
