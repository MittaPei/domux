#!/usr/bin/env python3
"""Replay the immutable v1 model outputs through the current Domux policy.

This is deliberately a post-formal, exploratory replay.  It does not run the
model, does not replace the v1 formal result, and does not emit confirmatory
inference.  The v1 manifest and the pre-implementation v2 policy plan are
pinned here so that a replay cannot silently change its source evidence or its
declared interpretation after results are known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import clarify_commit as policy_module
import evaluate as evaluator_module
from evaluate import (
    EvaluationInputError,
    FORMAL_BASE_COUNT,
    VARIANTS,
    _canonical_lines,
    _load_evidence,
    _sha256_bytes,
    _sha256_file,
    _validate_protocol,
    _validate_rows,
    _validated_model_run,
    canonical_json,
    run_evaluation,
)


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = CASE_DIR / "data" / "scenarios.jsonl"
DEFAULT_PROTOCOL = CASE_DIR / "data" / "protocol.json"
DEFAULT_FREEZE = CASE_DIR / "data" / "freeze.json"
DEFAULT_V1_DIR = CASE_DIR / "evidence" / "v1"
DEFAULT_V1_EVIDENCE = DEFAULT_V1_DIR / "domux_raw.jsonl"
DEFAULT_V1_METADATA = DEFAULT_V1_DIR / "model_metadata.json"
DEFAULT_V1_REPORT = DEFAULT_V1_DIR / "report.json"
DEFAULT_V1_TRIALS = DEFAULT_V1_DIR / "trials.jsonl"
DEFAULT_V1_MANIFEST = DEFAULT_V1_DIR / "manifest.json"
DEFAULT_V1_CODE_DIR = DEFAULT_V1_DIR / "code"
DEFAULT_POLICY_PLAN = CASE_DIR / "evidence" / "v2" / "policy_plan.json"
DEFAULT_CODE_FREEZE = CASE_DIR / "evidence" / "v2" / "code_freeze.json"
DEFAULT_OUTPUT_DIR = CASE_DIR / "evidence" / "v2"

# These inputs were frozen before the v2 implementation.  Pinning their bytes
# closes the otherwise circular trust chain in which a modified manifest,
# policy plan, protocol, or freeze file could bless modified evidence.
PINNED_V1_MANIFEST_SHA256 = "5f1c842676a367a9b5ae2cd948239a4f111bf0498e3cc916b57239ea671a9396"
PINNED_POLICY_PLAN_SHA256 = "b1727d6eb47367522a54ed1d1b5b1d200c8d5b9fbd30c7fd830de1a607f89302"
PINNED_PROTOCOL_SHA256 = "e0b1c0d1db4237311707401f469f5cb69a581a711fce6d6a314172a9449641e0"
PINNED_FREEZE_SHA256 = "01664c0f4b4f7dbc7eb6a239054927fc0c0889a74b95faef975298d3abe1a45c"

ARTIFACT_NAMES = ("trials.jsonl", "report.json")
COMPLETION_MARKER = "manifest.json"
OUTPUT_NAMES = (*ARTIFACT_NAMES, COMPLETION_MARKER)
HEX_DIGITS = frozenset("0123456789abcdef")
V1_CODE_NAMES = ("clarify_commit.py", "evaluate.py", "run_model.py")
CODE_FREEZE_FILES = (
    "clarify_commit.py",
    "evaluate.py",
    "ha_acceptance.py",
    "replay_policy.py",
    "run_model.py",
    "requirements.txt",
    "data/DATA_CARD.md",
    "evidence/v2/policy_plan.json",
    "tests/test_clarify_commit.py",
    "tests/test_dataset.py",
    "tests/test_evaluate.py",
    "tests/test_ha_acceptance.py",
    "tests/test_replay_policy.py",
    "tests/test_run_model.py",
)

# Bind the loaded Python modules to the source bytes visible at import time.
# replay_to_directory verifies these bytes again before and after evaluation.
IMPORTED_POLICY_SHA256 = _sha256_file(CASE_DIR / "clarify_commit.py")
IMPORTED_EVALUATOR_SHA256 = _sha256_file(CASE_DIR / "evaluate.py")
IMPORTED_REPLAY_SHA256 = _sha256_file(Path(__file__).resolve())


class ReplayInputError(EvaluationInputError):
    """The exploratory replay contract or an immutable input was violated."""


class ReplayPublicationError(ReplayInputError):
    """A complete or partial immutable publication already occupies the target."""


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise ReplayInputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_git_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise ReplayInputError(f"{label} must be a full lowercase Git commit ID")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ReplayInputError(f"{label} must be a JSON object")
    return value


def _read_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayInputError(f"cannot read {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReplayInputError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ReplayInputError(f"{label} changed while it was read")
    return payload


def _json_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"cannot read valid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ReplayInputError(f"{label} must be a JSON object")
    return value


def _jsonl_from_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReplayInputError(f"cannot read {label} JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayInputError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ReplayInputError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _sha256_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_execution_source_bindings() -> None:
    expected_paths = {
        "policy": CASE_DIR / "clarify_commit.py",
        "evaluator": CASE_DIR / "evaluate.py",
    }
    observed_paths = {
        "policy": Path(policy_module.__file__).resolve(),
        "evaluator": Path(evaluator_module.__file__).resolve(),
    }
    for label, expected in expected_paths.items():
        if observed_paths[label] != expected.resolve():
            raise ReplayInputError(f"loaded {label} module is outside the case directory")
    current = {
        "policy": _sha256_file(expected_paths["policy"]),
        "evaluator": _sha256_file(expected_paths["evaluator"]),
        "replay": _sha256_file(Path(__file__).resolve()),
    }
    imported = {
        "policy": IMPORTED_POLICY_SHA256,
        "evaluator": IMPORTED_EVALUATOR_SHA256,
        "replay": IMPORTED_REPLAY_SHA256,
    }
    for label, digest in current.items():
        if digest != imported[label]:
            raise ReplayInputError(f"{label} source changed after its module was loaded")
    if run_evaluation is not evaluator_module.run_evaluation:
        raise ReplayInputError("evaluation function does not match the loaded evaluator module")


def _code_bundle_sha256(files: Mapping[str, object]) -> str:
    return _sha256_payload(canonical_json({"files": files}).encode("utf-8"))


def _validate_code_freeze(
    manifest: Mapping[str, object],
    *,
    observed_sha256: str,
    captured_code: Mapping[str, bytes],
) -> dict[str, object]:
    """Validate the content-addressed v2 source bundle without Git history.

    GitHub may squash the contribution, so fork commit IDs are provenance only.
    The relative-path hashes are the authoritative, fresh-clone-stable binding.
    """

    if manifest.get("schema_version") != 1:
        raise ReplayInputError("v2 code-freeze schema changed")
    if manifest.get("manifest_type") != "domux-v2-code-freeze":
        raise ReplayInputError("v2 code-freeze manifest type changed")
    if manifest.get("evidence_version") != "v2-post-formal-exploratory":
        raise ReplayInputError("v2 code-freeze evidence version changed")
    if manifest.get("status") != "frozen-before-official-replay":
        raise ReplayInputError("v2 code-freeze status changed")
    if manifest.get("hash_algorithm") != "sha256":
        raise ReplayInputError("v2 code-freeze hash algorithm changed")
    if manifest.get("authority") != "content-addressed-source-bundle":
        raise ReplayInputError("v2 code-freeze authority changed")

    files = _require_mapping(manifest.get("files"), "v2 code-freeze files")
    if set(files) != set(CODE_FREEZE_FILES):
        raise ReplayInputError("v2 code-freeze file set changed")
    if set(captured_code) != set(CODE_FREEZE_FILES):
        raise ReplayInputError("captured v2 source file set is incomplete")
    for relative in CODE_FREEZE_FILES:
        binding = _require_mapping(files.get(relative), f"v2 source {relative}")
        if set(binding) != {"sha256", "size_bytes"}:
            raise ReplayInputError(f"v2 source binding shape changed: {relative}")
        expected_sha256 = _require_sha256(
            binding.get("sha256"), f"v2 source {relative}"
        )
        payload = captured_code[relative]
        if expected_sha256 != _sha256_payload(payload):
            raise ReplayInputError(f"v2 source hash mismatch: {relative}")
        if binding.get("size_bytes") != len(payload):
            raise ReplayInputError(f"v2 source size mismatch: {relative}")

    bundle_sha256 = _require_sha256(
        manifest.get("bundle_sha256"), "v2 code-freeze bundle"
    )
    if bundle_sha256 != _code_bundle_sha256(files):
        raise ReplayInputError("v2 code-freeze bundle hash mismatch")
    provenance = _require_mapping(
        manifest.get("fork_git_provenance"), "v2 fork Git provenance"
    )
    commit = _require_git_commit(
        provenance.get("source_commit"), "v2 fork source commit"
    )
    if provenance.get("role") != "informational-only; content hashes are authoritative":
        raise ReplayInputError("v2 fork provenance role changed")
    return {
        "manifest_sha256": observed_sha256,
        "bundle_sha256": bundle_sha256,
        "file_count": len(files),
        "authority": "content-addressed-source-bundle",
        "fork_source_commit": commit,
        "fork_commit_required_for_replay": False,
        "squash_merge_safe": True,
    }


def _validate_captured_files_unchanged(captured: Mapping[Path, bytes]) -> None:
    for path, payload in captured.items():
        if _sha256_file(path) != _sha256_payload(payload):
            raise ReplayInputError(f"input changed during replay: {path.name}")


def _validate_policy_plan(
    plan: Mapping[str, object],
    *,
    observed_sha256: str,
) -> Mapping[str, object]:
    if observed_sha256 != PINNED_POLICY_PLAN_SHA256:
        raise ReplayInputError("v2 policy plan differs from its pre-implementation bytes")
    if plan.get("schema_version") != 1:
        raise ReplayInputError("v2 policy plan schema changed")
    if plan.get("evidence_version") != "v2-post-formal-exploratory":
        raise ReplayInputError("v2 policy plan evidence classification changed")
    if plan.get("status") != "declared_before_v2_implementation":
        raise ReplayInputError("v2 policy plan is not the pre-implementation declaration")
    if plan.get("parent_v1_commit") != "fa33c10":
        raise ReplayInputError("v2 policy plan parent v1 commit changed")

    limits = plan.get("interpretation_limits")
    if not isinstance(limits, list) or any(not isinstance(item, str) for item in limits):
        raise ReplayInputError("v2 policy plan interpretation limits are malformed")
    required_limits = (
        "not a held-out or pre-registered confirmatory evaluation",
        "v1 remains the sole formal headline result",
        "must not report a new confirmatory p-value",
    )
    normalized = " ".join(limits).lower()
    if any(requirement not in normalized for requirement in required_limits):
        raise ReplayInputError("v2 policy plan lost a required interpretation limit")

    invariants = plan.get("policy_invariants")
    if not isinstance(invariants, list) or not invariants:
        raise ReplayInputError("v2 policy plan must declare policy invariants")
    invariant_ids = {
        str(item.get("id"))
        for item in invariants
        if isinstance(item, dict) and isinstance(item.get("rule"), str)
    }
    if invariant_ids != {
        "direction-evidence",
        "unregistered-selector",
        "deictic-context-first",
    }:
        raise ReplayInputError("v2 policy plan invariant set changed")
    return _require_mapping(plan.get("source_evidence"), "v2 source_evidence")


def _validate_v1_manifest(
    manifest: Mapping[str, object],
    *,
    observed_sha256: str,
    artifact_sha256: Mapping[str, str],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if observed_sha256 != PINNED_V1_MANIFEST_SHA256:
        raise ReplayInputError("v1 manifest differs from its frozen bytes")
    if manifest.get("schema_version") != 1:
        raise ReplayInputError("v1 manifest schema changed")
    if manifest.get("evidence_version") != "v1-formal":
        raise ReplayInputError("v1 manifest evidence version changed")
    if manifest.get("status") != "frozen":
        raise ReplayInputError("v1 manifest is not frozen")
    if manifest.get("role") != "sole pre-remediation formal evaluation":
        raise ReplayInputError("v1 manifest no longer declares the sole formal evaluation")

    artifacts = _require_mapping(manifest.get("artifacts"), "v1 artifacts")
    code = _require_mapping(manifest.get("code"), "v1 code")
    expected_artifacts = {
        "domux_raw.jsonl", "model_metadata.json", "report.json", "trials.jsonl",
    }
    expected_code = {"clarify_commit.py", "evaluate.py", "run_model.py"}
    if set(artifacts) != expected_artifacts or set(code) != expected_code:
        raise ReplayInputError("v1 manifest artifact/code key set changed")
    for name in expected_artifacts:
        _require_sha256(artifacts.get(name), f"v1 artifact {name}")
    for name in expected_code:
        _require_sha256(code.get(name), f"v1 code {name}")
    for name, observed in artifact_sha256.items():
        if artifacts[name] != observed:
            label = {
                "domux_raw.jsonl": "raw evidence",
                "model_metadata.json": "model metadata",
                "report.json": "report",
                "trials.jsonl": "trials",
            }[name]
            raise ReplayInputError(
                f"v1 {label} hash does not match the frozen manifest"
            )
    return artifacts, code


def _validate_v1_results(
    report: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
) -> None:
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ReplayInputError("v1 report is not a complete schema-v1 result")
    population = _require_mapping(report.get("population"), "v1 population")
    if population.get("evaluation_bases") != 48 or population.get("paired_probes") != 96:
        raise ReplayInputError("v1 report population denominator changed")
    trial_counts = _require_mapping(report.get("trial_counts"), "v1 trial counts")
    if trial_counts.get("total_records") != len(trials) or len(trials) != 936:
        raise ReplayInputError("v1 report/trials record count mismatch")


def _cross_validate_bindings(
    *,
    source_evidence: Mapping[str, object],
    artifacts: Mapping[str, object],
    code: Mapping[str, object],
    metadata: Mapping[str, object],
    evidence_sha256: str,
    metadata_sha256: str,
    evaluator_sha256: str,
    runner_sha256: str,
) -> None:
    expected_plan_bindings = {
        "raw_output_sha256": evidence_sha256,
        "model_metadata_sha256": metadata_sha256,
        "v1_policy_sha256": code["clarify_commit.py"],
        "v1_evaluator_sha256": code["evaluate.py"],
    }
    for field, expected in expected_plan_bindings.items():
        if source_evidence.get(field) != expected:
            raise ReplayInputError(f"v2 policy plan source binding mismatch: {field}")

    expected_metadata_bindings = {
        "raw_evidence_sha256": artifacts["domux_raw.jsonl"],
        "grounding_policy_sha256": code["clarify_commit.py"],
        "evaluator_sha256": code["evaluate.py"],
        "runner_sha256": code["run_model.py"],
    }
    for field, expected in expected_metadata_bindings.items():
        if metadata.get(field) != expected:
            raise ReplayInputError(f"v1 model metadata/code binding mismatch: {field}")

    # The revised policy is intentionally allowed to differ.  The evaluator
    # and runner are not: keeping them byte-identical preserves the v1
    # validator and model-evidence contract used by this replay.
    if evaluator_sha256 != code["evaluate.py"]:
        raise ReplayInputError("current evaluator differs from the v1-bound evaluator")
    if runner_sha256 != code["run_model.py"]:
        raise ReplayInputError("current model runner differs from the v1-bound runner")


def _validate_formal_metadata_with_revised_policy(
    *,
    metadata: Mapping[str, object],
    protocol: Mapping[str, object],
    dataset_sha256: str,
    evidence_sha256: str,
    evidence_failure_count: int,
    current_policy_sha256: str,
) -> dict[str, object]:
    """Reuse v1's validator while preserving the original metadata object.

    ``evaluate._validated_model_run`` normally compares the recorded policy
    hash to the policy currently imported beside it.  That is correct for a
    formal run and intentionally false for this remediation replay.  The
    original metadata bytes have already been pinned and cross-validated above;
    this in-memory copy changes only that one comparison so all remaining model,
    snapshot, generation, data, evidence, and runtime checks stay centralized
    in the unchanged v1 validator.
    """

    validator_view = dict(metadata)
    validator_view["grounding_policy_sha256"] = current_policy_sha256
    validated = _validated_model_run(
        validator_view,
        protocol,
        dataset_sha256,
        evidence_sha256,
        evidence_failure_count,
    )
    validated["artifact_origin"] = "v1-formal"
    validated["replayed_without_model_inference"] = True
    # The validator's synthetic current-policy binding is an implementation
    # detail.  Report the actual, immutable v1 bindings instead.
    validated.pop("code_binding", None)
    return validated


def _validate_freeze_from_captured_bytes(
    freeze: Mapping[str, object],
    *,
    dataset_sha256: str,
    protocol_sha256: str,
    snapshot_manifest_sha256: str,
    eval_rows: Sequence[Mapping[str, object]],
) -> None:
    checks = {
        "full_sha256": dataset_sha256,
        "protocol_sha256": protocol_sha256,
        "evaluation_sha256": _sha256_bytes(_canonical_lines(eval_rows)),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
    }
    for field, observed in checks.items():
        if freeze.get(field) != observed:
            raise ReplayInputError(f"frozen input hash mismatch: {field}")
    if freeze.get("evaluation_count") != FORMAL_BASE_COUNT:
        raise ReplayInputError("freeze evaluation denominator is not 48")


def _execute_evaluation(
    eval_rows: Sequence[dict[str, Any]],
    protocol: Mapping[str, object],
    evidence: Mapping[tuple[str, str], dict[str, Any]],
    *,
    integrity: Mapping[str, object],
    model_run: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return run_evaluation(
        eval_rows,
        protocol,
        evidence,
        integrity=integrity,
        model_run=model_run,
    )


def _output_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _refuse_existing_outputs(output_dir: Path) -> None:
    existing = [name for name in OUTPUT_NAMES if _output_exists(output_dir / name)]
    if existing:
        raise ReplayPublicationError(
            "refusing to overwrite existing replay output: " + ", ".join(existing)
        )


def _prepare_temp_file(output_dir: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".replay-policy-", dir=output_dir)
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path


def _validate_output_payloads(payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(OUTPUT_NAMES):
        raise ReplayInputError("v2 publication payload set is incomplete")
    manifest = _json_from_bytes(payloads[COMPLETION_MARKER], "v2 manifest")
    outputs = _require_mapping(manifest.get("outputs"), "v2 manifest outputs")
    if set(outputs) != set(ARTIFACT_NAMES):
        raise ReplayInputError("v2 manifest output set changed")
    for name in ARTIFACT_NAMES:
        binding = _require_mapping(outputs.get(name), f"v2 output {name}")
        if binding.get("sha256") != _sha256_payload(payloads[name]):
            raise ReplayInputError(f"v2 manifest hash mismatch: {name}")
        if binding.get("size_bytes") != len(payloads[name]):
            raise ReplayInputError(f"v2 manifest size mismatch: {name}")


def _publish_outputs_with_completion_marker(
    output_dir: Path,
    payloads: Mapping[str, bytes],
) -> None:
    """Verify artifacts first, then atomically expose the completion marker."""

    _validate_output_payloads(payloads)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)
    temporary: dict[str, Path] = {}
    created: list[Path] = []
    try:
        for name in OUTPUT_NAMES:
            temporary[name] = _prepare_temp_file(output_dir, payloads[name])
            if _read_bytes(temporary[name], f"temporary {name}") != payloads[name]:
                raise ReplayInputError(f"temporary {name} differs from generated bytes")
        for name in ARTIFACT_NAMES:
            destination = output_dir / name
            try:
                # A same-directory hard link is an atomic, no-overwrite publish.
                os.link(temporary[name], destination)
            except FileExistsError as exc:
                raise ReplayPublicationError(
                    f"refusing to overwrite existing replay output: {name}"
                ) from exc
            created.append(destination)
            temporary[name].unlink()
        # A manifest is a trustworthy completion marker only after both linked
        # artifacts have been read back and matched to its already-validated
        # bindings.  Any failure here removes the partial set without ever
        # exposing a marker.
        for name in ARTIFACT_NAMES:
            observed = _read_bytes(output_dir / name, f"published {name}")
            if observed != payloads[name]:
                raise ReplayInputError(f"published {name} differs from generated bytes")
        marker = output_dir / COMPLETION_MARKER
        try:
            os.link(temporary[COMPLETION_MARKER], marker)
        except FileExistsError as exc:
            raise ReplayPublicationError(
                f"refusing to overwrite existing replay output: {COMPLETION_MARKER}"
            ) from exc
        created.append(marker)
        temporary[COMPLETION_MARKER].unlink()
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def replay_to_directory(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    protocol_path: Path = DEFAULT_PROTOCOL,
    freeze_path: Path = DEFAULT_FREEZE,
    evidence_path: Path = DEFAULT_V1_EVIDENCE,
    metadata_path: Path = DEFAULT_V1_METADATA,
    v1_report_path: Path = DEFAULT_V1_REPORT,
    v1_trials_path: Path = DEFAULT_V1_TRIALS,
    manifest_path: Path = DEFAULT_V1_MANIFEST,
    v1_code_dir: Path = DEFAULT_V1_CODE_DIR,
    policy_plan_path: Path = DEFAULT_POLICY_PLAN,
    code_freeze_path: Path = DEFAULT_CODE_FREEZE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Validate frozen bytes, replay all 96 probes, and publish one v2 set."""

    _refuse_existing_outputs(output_dir)
    _validate_execution_source_bindings()

    paths = {
        "dataset": dataset_path,
        "protocol": protocol_path,
        "freeze": freeze_path,
        "v1_evidence": evidence_path,
        "v1_metadata": metadata_path,
        "v1_report": v1_report_path,
        "v1_trials": v1_trials_path,
        "v1_manifest": manifest_path,
        "v2_policy_plan": policy_plan_path,
        "v2_code_freeze": code_freeze_path,
        "runner": CASE_DIR / "run_model.py",
    }
    captured = {
        path: _read_bytes(path, label.replace("_", " "))
        for label, path in paths.items()
    }
    digests = {label: _sha256_payload(captured[path]) for label, path in paths.items()}
    evidence_sha256 = digests["v1_evidence"]
    metadata_sha256 = digests["v1_metadata"]
    manifest_sha256 = digests["v1_manifest"]
    policy_plan_sha256 = digests["v2_policy_plan"]
    current_policy_sha256 = IMPORTED_POLICY_SHA256
    evaluator_sha256 = IMPORTED_EVALUATOR_SHA256
    runner_sha256 = digests["runner"]
    replay_sha256 = IMPORTED_REPLAY_SHA256
    protocol_sha256 = digests["protocol"]
    freeze_sha256 = digests["freeze"]
    if protocol_sha256 != PINNED_PROTOCOL_SHA256:
        raise ReplayInputError("protocol differs from the v1 evaluation protocol")
    if freeze_sha256 != PINNED_FREEZE_SHA256:
        raise ReplayInputError("freeze manifest differs from the v1 evaluation freeze")
    snapshot_manifest_path = protocol_path.parent / "snapshot_manifest.json"
    snapshot_payload = _read_bytes(snapshot_manifest_path, "snapshot manifest")
    paths["snapshot_manifest"] = snapshot_manifest_path
    captured[snapshot_manifest_path] = snapshot_payload
    digests["snapshot_manifest"] = _sha256_payload(snapshot_payload)

    code_freeze_manifest = _json_from_bytes(
        captured[code_freeze_path], "v2 code freeze"
    )
    captured_code: dict[str, bytes] = {}
    for relative in CODE_FREEZE_FILES:
        path = CASE_DIR / relative
        payload = captured.get(path)
        if payload is None:
            payload = _read_bytes(path, f"v2 source {relative}")
            captured[path] = payload
        captured_code[relative] = payload
        label = f"v2_source:{relative}"
        paths[label] = path
        digests[label] = _sha256_payload(payload)
    freeze_binding = _validate_code_freeze(
        code_freeze_manifest,
        observed_sha256=digests["v2_code_freeze"],
        captured_code=captured_code,
    )

    manifest = _json_from_bytes(captured[manifest_path], "v1 manifest")
    policy_plan = _json_from_bytes(captured[policy_plan_path], "v2 policy plan")
    metadata = _json_from_bytes(captured[metadata_path], "v1 model-run metadata")
    v1_report = _json_from_bytes(captured[v1_report_path], "v1 report")
    v1_trials = _jsonl_from_bytes(captured[v1_trials_path], "v1 trials")
    source_evidence = _validate_policy_plan(
        policy_plan,
        observed_sha256=policy_plan_sha256,
    )
    v1_artifact_sha256 = {
        "domux_raw.jsonl": evidence_sha256,
        "model_metadata.json": metadata_sha256,
        "report.json": digests["v1_report"],
        "trials.jsonl": digests["v1_trials"],
    }
    artifacts, v1_code = _validate_v1_manifest(
        manifest,
        observed_sha256=manifest_sha256,
        artifact_sha256=v1_artifact_sha256,
    )
    _validate_v1_results(v1_report, v1_trials)
    for name in V1_CODE_NAMES:
        path = v1_code_dir / name
        payload = _read_bytes(path, f"archived v1 source {name}")
        if _sha256_payload(payload) != v1_code[name]:
            raise ReplayInputError(f"archived v1 source hash mismatch: {name}")
        captured[path] = payload
        label = f"v1_archived_source:{name}"
        paths[label] = path
        digests[label] = _sha256_payload(payload)
    _cross_validate_bindings(
        source_evidence=source_evidence,
        artifacts=artifacts,
        code=v1_code,
        metadata=metadata,
        evidence_sha256=evidence_sha256,
        metadata_sha256=metadata_sha256,
        evaluator_sha256=evaluator_sha256,
        runner_sha256=runner_sha256,
    )
    if current_policy_sha256 == v1_code["clarify_commit.py"]:
        raise ReplayInputError("v2 policy is byte-identical to v1; no remediation to replay")

    dataset_rows = _jsonl_from_bytes(captured[dataset_path], "dataset")
    protocol = _json_from_bytes(captured[protocol_path], "protocol")
    freeze = _json_from_bytes(captured[freeze_path], "freeze manifest")
    _validate_protocol(protocol)
    eval_rows = _validate_rows(dataset_rows)
    _validate_freeze_from_captured_bytes(
        freeze,
        dataset_sha256=digests["dataset"],
        protocol_sha256=protocol_sha256,
        snapshot_manifest_sha256=digests["snapshot_manifest"],
        eval_rows=eval_rows,
    )
    evidence_rows = _jsonl_from_bytes(captured[evidence_path], "v1 raw evidence")
    evidence = _load_evidence(evidence_rows, eval_rows)
    if len(evidence) != FORMAL_BASE_COUNT * len(VARIANTS):
        raise ReplayInputError("policy replay requires all 96 paired v1 probes")

    model_run = _validate_formal_metadata_with_revised_policy(
        metadata=metadata,
        protocol=protocol,
        dataset_sha256=digests["dataset"],
        evidence_sha256=evidence_sha256,
        evidence_failure_count=sum(item.get("status") != "ok" for item in evidence_rows),
        current_policy_sha256=current_policy_sha256,
    )
    code_binding = {
        "v1_grounding_policy_sha256": v1_code["clarify_commit.py"],
        "v1_evaluator_sha256": v1_code["evaluate.py"],
        "v1_runner_sha256": v1_code["run_model.py"],
        "current_grounding_policy_sha256": current_policy_sha256,
        "current_evaluator_sha256": evaluator_sha256,
        "current_replay_policy_sha256": replay_sha256,
        "current_policy_differs_from_v1": current_policy_sha256
        != v1_code["clarify_commit.py"],
        "current_evaluator_matches_v1": evaluator_sha256 == v1_code["evaluate.py"],
        "current_runner_matches_v1": runner_sha256 == v1_code["run_model.py"],
        "v1_cross_bindings_verified": True,
        "v1_archived_source_bindings_verified": True,
        "v2_code_freeze": freeze_binding,
    }
    model_run["code_binding"] = code_binding
    integrity = {
        "dataset_sha256": digests["dataset"],
        "evaluation_sha256": _sha256_bytes(_canonical_lines(eval_rows)),
        "protocol_sha256": protocol_sha256,
        "freeze_manifest_sha256": freeze_sha256,
        "evidence_sha256": evidence_sha256,
        "model_run_metadata_sha256": metadata_sha256,
        "v1_manifest_sha256": manifest_sha256,
        "v2_policy_plan_sha256": policy_plan_sha256,
        "v2_code_freeze_manifest_sha256": digests["v2_code_freeze"],
        "freeze_verified": True,
        "evidence_pairs_verified": FORMAL_BASE_COUNT * len(VARIANTS),
        "v1_artifacts_sha256": v1_artifact_sha256,
        "v1_all_artifact_bindings_verified": True,
        "code_binding": code_binding,
    }

    trials, report = _execute_evaluation(
        eval_rows,
        protocol,
        evidence,
        integrity=integrity,
        model_run=model_run,
    )
    # The unchanged evaluator computes the v1 confirmatory section internally.
    # It is intentionally discarded rather than relabelled after failure-driven
    # policy work on the same fixed cases.
    report.pop("primary_inference", None)
    formal_methods = _require_mapping(report.get("methods"), "evaluation methods")
    report["methods"] = {
        "evaluation_engine": "evaluate.run_evaluation (v1 evaluator, byte-identical)",
        "binary_intervals": formal_methods["binary_intervals"],
        "binary_intervals_role": "descriptive uncertainty summaries only",
        "trial_reset": formal_methods["trial_reset"],
        "pseudo_replication_guards": formal_methods["pseudo_replication_guards"],
        "confirmatory_p_values": "omitted",
        "latency_measurement": (
            "Reused immutable v1 model-inference latency; this policy replay "
            "performs no model inference and adds no new latency measurement."
        ),
    }
    evaluator_gate = _require_mapping(report.pop("quality_gate"), "evaluator gate")
    report["exploratory_gate"] = {
        "result": evaluator_gate["result"],
        "criterion": (
            "Every eligible B2 clean/guard trial passes the fixed v1 oracle under "
            "the revised policy."
        ),
        "interpretation": "descriptive remediation check; not a confirmatory result",
        "affects_process_exit": True,
    }
    report["schema_version"] = 2
    report["evidence_version"] = "v2-post-formal-exploratory"
    report["analysis_classification"] = {
        "stage": "post-formal exploratory remediation replay",
        "held_out": False,
        "pre_registered": False,
        "confirmatory": False,
        "formal_headline_version": "v1-formal",
        "v1_remains_sole_formal": True,
        "model_rerun": False,
        "reused_v1_raw_outputs": FORMAL_BASE_COUNT * len(VARIANTS),
        "selective_policy_replay": False,
        "recorded_evidence_publication_count": 1,
        "publication_count_scope": "one content-addressed official v2 record",
        "byte_identical_reproduction_is_new_record": False,
        "production_generalization_claimed": False,
    }
    report["determinism"].update({
        "no_overwrite_outputs": True,
        "completion_marker": COMPLETION_MARKER,
        "completion_rule": "trust the output set only when manifest hashes verify",
        "generation_file_mode": "0600; Git/fresh-clone modes may be 0644",
    })

    trials_payload = "".join(
        canonical_json(trial) + "\n" for trial in trials
    ).encode("utf-8")
    report["output_integrity"] = {
        "trials_sha256": _sha256_payload(trials_payload),
        "v2_policy_plan_sha256": policy_plan_sha256,
        "v2_code_freeze": freeze_binding,
        "completion_marker": COMPLETION_MARKER,
    }
    report_payload = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output_artifacts = {
        "trials.jsonl": {
            "sha256": _sha256_payload(trials_payload),
            "size_bytes": len(trials_payload),
            "record_count": len(trials),
        },
        "report.json": {
            "sha256": _sha256_payload(report_payload),
            "size_bytes": len(report_payload),
        },
    }
    output_manifest = {
        "schema_version": 1,
        "manifest_type": "domux-v2-policy-replay-publication",
        "evidence_version": "v2-post-formal-exploratory",
        "status": "complete",
        "publication": {
            "official_recorded_ordinal": 1,
            "completion_marker": COMPLETION_MARKER,
            "marker_written_last": True,
            "no_overwrite": True,
            "v1_remains_sole_formal": True,
            "record_identity": "content-addressed",
            "byte_identical_reproduction_is_new_record": False,
        },
        "source_binding": freeze_binding,
        "inputs": {
            label: {"sha256": digests[label], "size_bytes": len(captured[path])}
            for label, path in paths.items()
            if label != "runner"
        },
        "code_binding": {
            "executed_code": {
                "clarify_commit.py": current_policy_sha256,
                "evaluate.py": evaluator_sha256,
                "replay_policy.py": replay_sha256,
            },
            "provenance_only_code": {"run_model.py": runner_sha256},
            "loaded_module_paths_verified": True,
            "pre_post_hashes_match": True,
            "content_addressed_bundle_verified": True,
            "requires_fork_git_history": False,
        },
        "outputs": output_artifacts,
        "result": {
            "analysis": "post-formal exploratory remediation replay",
            "model_rerun": False,
            "confirmatory": False,
            "exploratory_gate": report["exploratory_gate"]["result"],
        },
        "generation": {
            "file_mode_at_generation": "0600",
            "git_or_fresh_clone_mode_may_be": "0644",
        },
    }
    manifest_payload = (
        json.dumps(output_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    payloads = {
        "trials.jsonl": trials_payload,
        "report.json": report_payload,
        COMPLETION_MARKER: manifest_payload,
    }

    _validate_execution_source_bindings()
    _validate_captured_files_unchanged(captured)
    _publish_outputs_with_completion_marker(output_dir, payloads)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v1-evidence", type=Path, default=DEFAULT_V1_EVIDENCE)
    parser.add_argument("--v1-metadata", type=Path, default=DEFAULT_V1_METADATA)
    parser.add_argument("--v1-report", type=Path, default=DEFAULT_V1_REPORT)
    parser.add_argument("--v1-trials", type=Path, default=DEFAULT_V1_TRIALS)
    parser.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    parser.add_argument("--v1-code-dir", type=Path, default=DEFAULT_V1_CODE_DIR)
    parser.add_argument("--policy-plan", type=Path, default=DEFAULT_POLICY_PLAN)
    parser.add_argument("--code-freeze", type=Path, default=DEFAULT_CODE_FREEZE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args(argv)
    try:
        report = replay_to_directory(
            dataset_path=args.dataset,
            protocol_path=args.protocol,
            freeze_path=args.freeze,
            evidence_path=args.v1_evidence,
            metadata_path=args.v1_metadata,
            v1_report_path=args.v1_report,
            v1_trials_path=args.v1_trials,
            manifest_path=args.v1_manifest,
            v1_code_dir=args.v1_code_dir,
            policy_plan_path=args.policy_plan,
            code_freeze_path=args.code_freeze,
            output_dir=args.output_dir,
        )
    except ReplayPublicationError as exc:
        print(canonical_json({"status": "publication_conflict", "reason": str(exc)}))
        return 3
    except EvaluationInputError as exc:
        print(canonical_json({"status": "error", "reason": str(exc)}))
        return 2
    except OSError as exc:
        print(canonical_json({"status": "runtime_error", "reason": str(exc)}))
        return 1
    print(canonical_json({
        "status": report["status"],
        "analysis": report["analysis_classification"]["stage"],
        "exploratory_gate": report["exploratory_gate"]["result"],
        "evaluation_bases": report["population"]["evaluation_bases"],
        "trial_records": report["trial_counts"]["total_records"],
    }))
    return 0 if report["exploratory_gate"]["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
