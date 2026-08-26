#!/usr/bin/env python3
"""Offline verifier for the immutable Domux case evidence.

The verifier intentionally uses only the Python standard library.  It checks
the frozen v1 model run, the post-formal v2 replay publication, and the pinned
Home Assistant acceptance record.  It also binds the post-v2 v3 hardening plan,
validation record, policy, and regression tests without executing the model,
policy replay, Docker, Home Assistant, or any network operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


CASE_DIR = Path(__file__).resolve().parent

PINNED_V1_MANIFEST_SHA256 = (
    "5f1c842676a367a9b5ae2cd948239a4f111bf0498e3cc916b57239ea671a9396"
)
PINNED_V2_POLICY_PLAN_SHA256 = (
    "b1727d6eb47367522a54ed1d1b5b1d200c8d5b9fbd30c7fd830de1a607f89302"
)
PINNED_V2_CODE_FREEZE_SHA256 = (
    "f5f7f97ce5bd2d74dacce1b10457a3940e3f95655464cf8bc0368c04ca7dec4b"
)
PINNED_V2_MANIFEST_SHA256 = (
    "53feda5739aacea28f709e7882f6ca35cdc7c285acddc64de57c78ff54559c08"
)
PINNED_HA_ACCEPTANCE_SHA256 = (
    "fc3132b74978e3bb73954a681800cc13b398dd57267a0be953c83fd23e40d1e7"
)
PINNED_V3_POLICY_PLAN_SHA256 = (
    "025a5ed08416bb3b41f6fcd5ecb90e769cc47ead46eb9caf863a6f3b467895d4"
)
PINNED_V3_VALIDATION_SHA256 = (
    "da2d5d3fef495de1196e95ac456bce7733068de40fbf2ec971eee4c3266184f0"
)
PINNED_V3_HARDENING_COMMIT = "482b94eea78ac198f2abbfac5f2f16da02fb7b9e"

FORMAL_BASE_COUNT = 48
CONTEXT_BASE_COUNT = 12
VARIANTS = ("clear", "ambiguous")
ARMS = (
    "B0_unique_or_abstain",
    "B1_clarify_and_prepare",
    "B2_clarify_and_commit",
)
MUTATIONS = (
    "clean",
    "replay",
    "expiry",
    "target_drift",
    "session_swap",
    "plan_swap",
    "candidate_change",
    "context_state_change",
    "unrelated_state_change",
)
UNIVERSAL_GUARD_MUTATIONS = (
    "replay",
    "expiry",
    "target_drift",
    "session_swap",
    "plan_swap",
    "candidate_change",
    "unrelated_state_change",
)
B1_EXPECTED_LIMITATIONS = (
    "replay",
    "expiry",
    "target_drift",
    "candidate_change",
    "context_state_change",
)
V2_CODE_FREEZE_FILES = (
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
V2_ARCHIVED_SOURCE_FILES = (
    "clarify_commit.py",
    "tests/test_clarify_commit.py",
)
REPLAY_OUTPUT_NAMES = ("trials.jsonl", "report.json", "manifest.json")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")


class VerificationError(ValueError):
    """An artifact does not satisfy the immutable evidence contract."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"cannot read {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError(f"{label} is not a regular file: {path}")
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
        raise VerificationError(f"{label} changed while it was read: {path}")
    return payload


def _load_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object")
    return value


def _load_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{label} is not valid UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise VerificationError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise VerificationError(f"{label} must be an array")
    return value


def _require(condition: object, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _require_digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and HEX_64.fullmatch(value), f"{label} is not SHA-256")
    return str(value)


def _read_pinned(path: Path, expected: str, label: str) -> bytes:
    payload = _read_bytes(path, label)
    _require(_sha256(payload) == expected, f"{label} hash mismatch")
    return payload


def _verify_binding(payload: bytes, binding: object, label: str) -> None:
    value = _mapping(binding, f"{label} binding")
    expected_hash = _require_digest(value.get("sha256"), f"{label} hash")
    _require(_sha256(payload) == expected_hash, f"{label} hash mismatch")
    _require(value.get("size_bytes") == len(payload), f"{label} size mismatch")


def _canonical_lines(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _verify_frozen_data(case_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = case_dir / "data"
    dataset_payload = _read_bytes(data_dir / "scenarios.jsonl", "frozen dataset")
    protocol_payload = _read_bytes(data_dir / "protocol.json", "frozen protocol")
    snapshot_payload = _read_bytes(
        data_dir / "snapshot_manifest.json", "snapshot manifest"
    )
    freeze = _load_json(
        _read_bytes(data_dir / "freeze.json", "freeze manifest"),
        "freeze manifest",
    )
    rows = _load_jsonl(dataset_payload, "frozen dataset")
    development = [row for row in rows if row.get("split") == "dev"]
    evaluation = [row for row in rows if row.get("split") == "eval"]
    _require(len(development) == 16, "development row count is not 16")
    _require(len(evaluation) == FORMAL_BASE_COUNT, "evaluation row count is not 48")
    base_ids = [row.get("base_id") for row in rows]
    _require(
        all(isinstance(base_id, str) and base_id for base_id in base_ids),
        "dataset contains an invalid base_id",
    )
    _require(len(set(base_ids)) == len(base_ids), "dataset contains duplicate base_id values")
    _require(freeze.get("status") == "pre_frozen_before_any_model_output", "freeze status changed")
    _require(freeze.get("full_sha256") == _sha256(dataset_payload), "dataset/freeze hash mismatch")
    _require(freeze.get("protocol_sha256") == _sha256(protocol_payload), "protocol/freeze hash mismatch")
    _require(
        freeze.get("snapshot_manifest_sha256") == _sha256(snapshot_payload),
        "snapshot-manifest/freeze hash mismatch",
    )
    _require(
        freeze.get("development_sha256") == _sha256(_canonical_lines(development)),
        "development split hash mismatch",
    )
    _require(
        freeze.get("evaluation_sha256") == _sha256(_canonical_lines(evaluation)),
        "evaluation split hash mismatch",
    )
    categories = Counter(str(row.get("category")) for row in evaluation)
    _require(dict(categories) == freeze.get("evaluation_category_counts"), "evaluation category counts changed")
    return rows, evaluation


def _metric(
    report: Mapping[str, object],
    path: Sequence[str],
    successes: int,
    denominator: int,
) -> None:
    value: object = report
    for component in path:
        value = _mapping(value, ".".join(path)).get(component)
    metric = _mapping(value, ".".join(path))
    label = ".".join(path)
    _require(metric.get("successes") == successes, f"{label} successes mismatch")
    _require(metric.get("denominator") == denominator, f"{label} denominator mismatch")
    expected_rate = successes / denominator
    observed_rate = metric.get("rate")
    _require(
        isinstance(observed_rate, (int, float))
        and not isinstance(observed_rate, bool)
        and math.isclose(float(observed_rate), expected_rate, rel_tol=0.0, abs_tol=1e-14),
        f"{label} rate mismatch",
    )


def _index_trials(
    trials: Sequence[Mapping[str, object]],
    evaluation: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]],
    dict[tuple[str, str, str], Mapping[str, object]],
]:
    eval_by_id = {str(row["base_id"]): row for row in evaluation}
    language: dict[tuple[str, str], Mapping[str, object]] = {}
    execution: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for trial in trials:
        base_id = str(trial.get("base_id", ""))
        _require(base_id in eval_by_id, f"trial contains unknown base_id: {base_id}")
        record_type = trial.get("record_type")
        if record_type == "language_probe":
            key = (base_id, str(trial.get("variant", "")))
            _require(key not in language, f"duplicate language trial: {key}")
            language[key] = trial
        elif record_type == "execution_trial":
            key = (
                base_id,
                str(trial.get("arm", "")),
                str(trial.get("mutation", "")),
            )
            _require(key not in execution, f"duplicate execution trial: {key}")
            execution[key] = trial
        else:
            raise VerificationError(f"unknown trial record_type: {record_type}")

    expected_language = {
        (base_id, variant) for base_id in eval_by_id for variant in VARIANTS
    }
    expected_execution: set[tuple[str, str, str]] = set()
    for base_id, row in eval_by_id.items():
        expected_execution.add((base_id, ARMS[0], "clean"))
        for arm in ARMS[1:]:
            for mutation in MUTATIONS:
                if mutation != "context_state_change" or row.get("category") == "context_reference":
                    expected_execution.add((base_id, arm, mutation))
    _require(set(language) == expected_language, "language trial key set is incomplete or changed")
    _require(set(execution) == expected_execution, "execution trial key set is incomplete or changed")
    return language, execution


def _verify_aggregates(
    report: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
    evaluation: Sequence[Mapping[str, object]],
) -> None:
    language, execution = _index_trials(trials, evaluation)
    base_ids = [str(row["base_id"]) for row in evaluation]
    context_ids = [
        str(row["base_id"])
        for row in evaluation
        if row.get("category") == "context_reference"
    ]
    counts = _mapping(report.get("trial_counts"), "trial_counts")
    _require(counts.get("language_probe_records") == len(language) == 96, "language trial count mismatch")
    _require(counts.get("execution_trial_records") == len(execution) == 840, "execution trial count mismatch")
    _require(counts.get("total_records") == len(trials) == 936, "total trial count mismatch")
    population = _mapping(report.get("population"), "population")
    _require(population.get("evaluation_bases") == FORMAL_BASE_COUNT, "population base count mismatch")
    _require(population.get("paired_probes") == 96, "population probe count mismatch")
    _require(population.get("context_guard_bases") == CONTEXT_BASE_COUNT, "context denominator mismatch")

    ambiguous = {base_id: language[(base_id, "ambiguous")] for base_id in base_ids}
    clear = {base_id: language[(base_id, "clear")] for base_id in base_ids}
    language_path = ("metrics", "language")
    _metric(
        report,
        (*language_path, "sensitivity"),
        sum(item.get("observed_disposition") == "clarify" for item in ambiguous.values()),
        FORMAL_BASE_COUNT,
    )
    _metric(
        report,
        (*language_path, "specificity"),
        sum(item.get("observed_disposition") == "unique" for item in clear.values()),
        FORMAL_BASE_COUNT,
    )
    _metric(
        report,
        (*language_path, "false_clarification_rate"),
        sum(item.get("observed_disposition") == "clarify" for item in clear.values()),
        FORMAL_BASE_COUNT,
    )
    _metric(
        report,
        (*language_path, "paired_discrimination"),
        sum(
            ambiguous[base_id].get("observed_disposition") == "clarify"
            and clear[base_id].get("observed_disposition") == "unique"
            for base_id in base_ids
        ),
        FORMAL_BASE_COUNT,
    )
    _metric(
        report,
        (*language_path, "candidate_coverage"),
        sum(item.get("candidate_coverage") is True for item in ambiguous.values()),
        FORMAL_BASE_COUNT,
    )
    _metric(
        report,
        (*language_path, "inference_error_rate"),
        sum(
            any(language[(base_id, variant)].get("evidence_status") != "ok" for variant in VARIANTS)
            for base_id in base_ids
        ),
        FORMAL_BASE_COUNT,
    )

    for key, trial in language.items():
        candidates = trial.get("candidate_ids")
        _require(isinstance(candidates, list), f"language candidate_ids malformed: {key}")
        _require(trial.get("candidate_count") == len(candidates), f"language candidate_count mismatch: {key}")
        gold = "unique" if key[1] == "clear" else "clarify"
        _require(trial.get("gold_disposition") == gold, f"language gold disposition mismatch: {key}")
        _require(trial.get("correct") is (trial.get("observed_disposition") == gold), f"language correctness mismatch: {key}")

    b2_completion = sum(
        execution[(base_id, ARMS[2], "clean")].get("exact_delta_success") is True
        for base_id in base_ids
    )
    _metric(
        report,
        (*language_path, "clarification_completion"),
        b2_completion,
        FORMAL_BASE_COUNT,
    )

    for arm in ARMS:
        clean = [execution[(base_id, arm, "clean")] for base_id in base_ids]
        prefix = ("metrics", "execution", arm)
        _metric(
            report,
            (*prefix, "ambiguous_clean_exact_delta_success"),
            sum(item.get("exact_delta_success") is True for item in clean),
            FORMAL_BASE_COUNT,
        )
        _metric(
            report,
            (*prefix, "dispatch_coverage"),
            sum(item.get("sut_calls") == 1 and item.get("dispatched") is True for item in clean),
            FORMAL_BASE_COUNT,
        )
        _metric(
            report,
            (*prefix, "wrong_target_transition_rate"),
            sum(item.get("wrong_target_transition") is True for item in clean),
            FORMAL_BASE_COUNT,
        )
        _metric(
            report,
            (*prefix, "zero_preconfirm_calls"),
            sum(item.get("preconfirm_sut_calls") == 0 for item in clean),
            FORMAL_BASE_COUNT,
        )
        if arm == ARMS[0]:
            _metric(
                report,
                (*prefix, "safe_abstention_rate"),
                sum(item.get("safe_abstention") is True for item in clean),
                FORMAL_BASE_COUNT,
            )
            continue
        for mutation in MUTATIONS:
            eligible = context_ids if mutation == "context_state_change" else base_ids
            _metric(
                report,
                (*prefix, "mutation_oracle_rates", mutation),
                sum(execution[(base_id, arm, mutation)].get("oracle_pass") is True for base_id in eligible),
                len(eligible),
            )
        universal_successes = sum(
            all(
                execution[(base_id, arm, mutation)].get("oracle_pass") is True
                for mutation in UNIVERSAL_GUARD_MUTATIONS
            )
            for base_id in base_ids
        )
        _metric(report, (*prefix, "universal_guard_rate"), universal_successes, FORMAL_BASE_COUNT)
        _metric(
            report,
            (*prefix, "context_guard_rate"),
            sum(
                execution[(base_id, arm, "context_state_change")].get("oracle_pass") is True
                for base_id in context_ids
            ),
            CONTEXT_BASE_COUNT,
        )

    interpretation = _mapping(report.get("interpretation"), "interpretation")
    unexpected = _mapping(interpretation.get("unexpected_failures"), "unexpected_failures")
    b2_failures = _mapping(unexpected.get("B2_full_suite"), "B2_full_suite")
    for mutation in MUTATIONS:
        eligible = context_ids if mutation == "context_state_change" else base_ids
        observed = sum(
            execution[(base_id, ARMS[2], mutation)].get("oracle_pass") is not True
            for base_id in eligible
        )
        _require(b2_failures.get(mutation) == observed, f"B2 failure count mismatch: {mutation}")
    _require(
        interpretation.get("b2_suite_pass") is (not any(b2_failures.values())),
        "B2 suite interpretation mismatch",
    )
    expected = _mapping(interpretation.get("expected_baseline_outcomes"), "expected baselines")
    _require(
        expected.get("B0_safe_abstentions")
        == sum(execution[(base_id, ARMS[0], "clean")].get("safe_abstention") is True for base_id in base_ids),
        "B0 abstention interpretation mismatch",
    )
    b1_limits = _mapping(expected.get("B1_deliberately_missing_guard_failures"), "B1 limitations")
    for mutation in B1_EXPECTED_LIMITATIONS:
        eligible = context_ids if mutation == "context_state_change" else base_ids
        value = _mapping(b1_limits.get(mutation), f"B1 limitation {mutation}")
        _require(value.get("eligible_bases") == len(eligible), f"B1 eligible count mismatch: {mutation}")
        _require(
            value.get("observed_expected_limitations")
            == sum(
                execution[(base_id, ARMS[1], mutation)].get("interpretation")
                == "expected_b1_guard_limitation"
                for base_id in eligible
            ),
            f"B1 limitation count mismatch: {mutation}",
        )


def _verify_v1(case_dir: Path, evaluation: Sequence[Mapping[str, object]]) -> dict[str, object]:
    v1_dir = case_dir / "evidence" / "v1"
    manifest_payload = _read_pinned(
        v1_dir / "manifest.json", PINNED_V1_MANIFEST_SHA256, "v1 manifest"
    )
    manifest = _load_json(manifest_payload, "v1 manifest")
    _require(manifest.get("schema_version") == 1, "v1 manifest schema changed")
    _require(manifest.get("evidence_version") == "v1-formal", "v1 evidence version changed")
    _require(manifest.get("status") == "frozen", "v1 manifest is not frozen")
    _require(manifest.get("role") == "sole pre-remediation formal evaluation", "v1 role changed")
    artifacts = _mapping(manifest.get("artifacts"), "v1 artifacts")
    code = _mapping(manifest.get("code"), "v1 code")
    expected_artifacts = {
        "domux_raw.jsonl",
        "model_metadata.json",
        "report.json",
        "trials.jsonl",
    }
    _require(set(artifacts) == expected_artifacts, "v1 artifact set changed")
    _require(set(code) == {"clarify_commit.py", "evaluate.py", "run_model.py"}, "v1 code set changed")
    payloads: dict[str, bytes] = {}
    for name in expected_artifacts:
        expected_hash = _require_digest(artifacts.get(name), f"v1 {name}")
        payloads[name] = _read_pinned(v1_dir / name, expected_hash, f"v1 {name}")
    for name, digest in code.items():
        expected_hash = _require_digest(digest, f"v1 code {name}")
        _read_pinned(v1_dir / "code" / name, expected_hash, f"archived v1 {name}")

    metadata = _load_json(payloads["model_metadata.json"], "v1 metadata")
    _require(metadata.get("raw_evidence_sha256") == artifacts["domux_raw.jsonl"], "v1 metadata/raw binding mismatch")
    _require(metadata.get("grounding_policy_sha256") == code["clarify_commit.py"], "v1 metadata/policy binding mismatch")
    _require(metadata.get("evaluator_sha256") == code["evaluate.py"], "v1 metadata/evaluator binding mismatch")
    _require(metadata.get("runner_sha256") == code["run_model.py"], "v1 metadata/runner binding mismatch")
    _require(metadata.get("sample_count") == 96 and metadata.get("base_count") == 48, "v1 model denominator changed")
    _require(metadata.get("sample_failures") == 0, "v1 model run contains failures")
    _require(metadata.get("selective_reruns") == 0, "v1 model run contains selective reruns")

    raw = _load_jsonl(payloads["domux_raw.jsonl"], "v1 raw evidence")
    expected_commands = {
        (str(row["base_id"]), variant): row[f"{variant}_command"]
        for row in evaluation
        for variant in VARIANTS
    }
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in raw:
        key = (str(item.get("base_id", "")), str(item.get("variant", "")))
        _require(key in expected_commands, f"v1 raw evidence contains unknown key: {key}")
        _require(key not in indexed, f"v1 raw evidence contains duplicate key: {key}")
        _require(item.get("command") == expected_commands[key], f"v1 raw command mismatch: {key}")
        _require(item.get("status") in {"ok", "error"}, f"v1 raw status invalid: {key}")
        raw_output = item.get("raw_output", "")
        _require(isinstance(raw_output, str), f"v1 raw output is not text: {key}")
        if "raw_output_sha256" in item:
            _require(item["raw_output_sha256"] == _sha256(raw_output.encode()), f"v1 raw output hash mismatch: {key}")
        indexed[key] = item
    _require(set(indexed) == set(expected_commands), "v1 raw evidence is incomplete")

    report = _load_json(payloads["report.json"], "v1 report")
    trials = _load_jsonl(payloads["trials.jsonl"], "v1 trials")
    _require(report.get("schema_version") == 1 and report.get("status") == "complete", "v1 report is incomplete")
    integrity = _mapping(report.get("input_integrity"), "v1 input integrity")
    expected_integrity = {
        "dataset_sha256": _sha256(
            _read_bytes(case_dir / "data" / "scenarios.jsonl", "frozen dataset")
        ),
        "evaluation_sha256": _sha256(_canonical_lines(evaluation)),
        "protocol_sha256": _sha256(
            _read_bytes(case_dir / "data" / "protocol.json", "frozen protocol")
        ),
        "evidence_sha256": artifacts["domux_raw.jsonl"],
        "model_run_metadata_sha256": artifacts["model_metadata.json"],
    }
    for field, expected in expected_integrity.items():
        _require(integrity.get(field) == expected, f"v1 report binding mismatch: {field}")
    _require(integrity.get("freeze_verified") is True, "v1 report did not verify the freeze")
    _require(integrity.get("evidence_pairs_verified") == 96, "v1 report evidence count changed")
    report_code = _mapping(integrity.get("code_binding"), "v1 report code binding")
    _require(report_code.get("match") is True, "v1 report code binding is not verified")
    for field, name in (
        ("grounding_policy_sha256", "clarify_commit.py"),
        ("evaluator_sha256", "evaluate.py"),
        ("runner_sha256", "run_model.py"),
    ):
        _require(report_code.get(field) == code[name], f"v1 report code mismatch: {name}")
    _verify_aggregates(report, trials, evaluation)
    return {
        "status": "verified",
        "manifest_sha256": PINNED_V1_MANIFEST_SHA256,
        "raw_probes": len(raw),
        "trial_records": len(trials),
        "formal_headline": True,
    }


def _verify_code_freeze(case_dir: Path) -> tuple[dict[str, Any], bytes]:
    path = case_dir / "evidence" / "v2" / "code_freeze.json"
    payload = _read_pinned(path, PINNED_V2_CODE_FREEZE_SHA256, "v2 code freeze")
    freeze = _load_json(payload, "v2 code freeze")
    _require(freeze.get("schema_version") == 1, "v2 code-freeze schema changed")
    _require(freeze.get("manifest_type") == "domux-v2-code-freeze", "v2 code-freeze type changed")
    _require(freeze.get("status") == "frozen-before-official-replay", "v2 code freeze status changed")
    _require(freeze.get("authority") == "content-addressed-source-bundle", "v2 code-freeze authority changed")
    files = _mapping(freeze.get("files"), "v2 code-freeze files")
    _require(set(files) == set(V2_CODE_FREEZE_FILES), "v2 code-freeze file set changed")
    for relative in V2_CODE_FREEZE_FILES:
        source_path = (
            case_dir / "evidence" / "v2" / "code" / relative
            if relative in V2_ARCHIVED_SOURCE_FILES
            else case_dir / relative
        )
        source = _read_bytes(source_path, f"v2 source {relative}")
        _verify_binding(source, files[relative], f"v2 source {relative}")
    bundle = _require_digest(freeze.get("bundle_sha256"), "v2 code-freeze bundle")
    _require(bundle == _sha256(canonical_json({"files": files}).encode()), "v2 code-freeze bundle mismatch")
    provenance = _mapping(freeze.get("fork_git_provenance"), "v2 fork provenance")
    commit = provenance.get("source_commit")
    _require(isinstance(commit, str) and HEX_40.fullmatch(commit), "v2 fork provenance commit is invalid")
    _require(
        provenance.get("role") == "informational-only; content hashes are authoritative",
        "v2 fork provenance role changed",
    )
    return freeze, payload


def _verify_v2(
    case_dir: Path,
    evaluation: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    v1_dir = case_dir / "evidence" / "v1"
    v2_dir = case_dir / "evidence" / "v2"
    policy_plan_payload = _read_pinned(
        v2_dir / "policy_plan.json",
        PINNED_V2_POLICY_PLAN_SHA256,
        "v2 policy plan",
    )
    policy_plan = _load_json(policy_plan_payload, "v2 policy plan")
    _require(policy_plan.get("status") == "declared_before_v2_implementation", "v2 policy plan status changed")
    freeze, freeze_payload = _verify_code_freeze(case_dir)
    manifest_payload = _read_pinned(
        v2_dir / "manifest.json", PINNED_V2_MANIFEST_SHA256, "v2 publication manifest"
    )
    manifest = _load_json(manifest_payload, "v2 publication manifest")
    _require(manifest.get("schema_version") == 1, "v2 publication schema changed")
    _require(manifest.get("manifest_type") == "domux-v2-policy-replay-publication", "v2 publication type changed")
    _require(manifest.get("status") == "complete", "v2 publication is incomplete")
    _require(manifest.get("evidence_version") == "v2-post-formal-exploratory", "v2 evidence version changed")
    publication = _mapping(manifest.get("publication"), "v2 publication")
    _require(publication.get("completion_marker") == "manifest.json", "v2 completion marker changed")
    _require(publication.get("marker_written_last") is True, "v2 manifest was not declared last")
    _require(publication.get("official_recorded_ordinal") == 1, "v2 publication ordinal changed")
    _require(publication.get("v1_remains_sole_formal") is True, "v2 displaced the formal v1 result")
    source_binding = _mapping(manifest.get("source_binding"), "v2 source binding")
    _require(source_binding.get("manifest_sha256") == _sha256(freeze_payload), "v2 manifest/code-freeze binding mismatch")
    _require(source_binding.get("bundle_sha256") == freeze.get("bundle_sha256"), "v2 source bundle binding mismatch")
    _require(source_binding.get("file_count") == len(V2_CODE_FREEZE_FILES), "v2 source file count mismatch")

    outputs = _mapping(manifest.get("outputs"), "v2 outputs")
    _require(set(outputs) == {"report.json", "trials.jsonl"}, "v2 output set changed")
    report_payload = _read_bytes(v2_dir / "report.json", "v2 report")
    trials_payload = _read_bytes(v2_dir / "trials.jsonl", "v2 trials")
    _verify_binding(report_payload, outputs["report.json"], "v2 report")
    _verify_binding(trials_payload, outputs["trials.jsonl"], "v2 trials")
    _require(
        _mapping(outputs["trials.jsonl"], "v2 trials binding").get("record_count") == 936,
        "v2 trials manifest count changed",
    )

    v1_names = {
        "v1_evidence": v1_dir / "domux_raw.jsonl",
        "v1_metadata": v1_dir / "model_metadata.json",
        "v1_report": v1_dir / "report.json",
        "v1_trials": v1_dir / "trials.jsonl",
        "v1_manifest": v1_dir / "manifest.json",
    }
    input_paths: dict[str, Path] = {
        "dataset": case_dir / "data" / "scenarios.jsonl",
        "protocol": case_dir / "data" / "protocol.json",
        "freeze": case_dir / "data" / "freeze.json",
        "snapshot_manifest": case_dir / "data" / "snapshot_manifest.json",
        "v2_policy_plan": v2_dir / "policy_plan.json",
        "v2_code_freeze": v2_dir / "code_freeze.json",
        **v1_names,
    }
    for name in ("clarify_commit.py", "evaluate.py", "run_model.py"):
        input_paths[f"v1_archived_source:{name}"] = v1_dir / "code" / name
    for relative in V2_CODE_FREEZE_FILES:
        input_paths[f"v2_source:{relative}"] = (
            v2_dir / "code" / relative
            if relative in V2_ARCHIVED_SOURCE_FILES
            else case_dir / relative
        )
    inputs = _mapping(manifest.get("inputs"), "v2 inputs")
    _require(set(inputs) == set(input_paths), "v2 publication input set changed")
    for label, path in input_paths.items():
        _verify_binding(_read_bytes(path, label), inputs[label], label)

    manifest_code = _mapping(manifest.get("code_binding"), "v2 manifest code binding")
    executed = _mapping(manifest_code.get("executed_code"), "v2 executed code")
    frozen_files = _mapping(freeze.get("files"), "v2 frozen files")
    for name in ("clarify_commit.py", "evaluate.py", "replay_policy.py"):
        expected = _mapping(frozen_files[name], f"v2 frozen {name}").get("sha256")
        _require(executed.get(name) == expected, f"v2 executed-code binding mismatch: {name}")
    provenance_code = _mapping(
        manifest_code.get("provenance_only_code"), "v2 provenance-only code"
    )
    _require(
        provenance_code.get("run_model.py")
        == _mapping(frozen_files["run_model.py"], "v2 frozen runner").get("sha256"),
        "v2 runner provenance binding mismatch",
    )
    _require(manifest_code.get("loaded_module_paths_verified") is True, "v2 module paths were not verified")
    _require(manifest_code.get("pre_post_hashes_match") is True, "v2 pre/post source hashes differ")
    _require(manifest_code.get("content_addressed_bundle_verified") is True, "v2 source bundle was not verified")
    _require(manifest_code.get("requires_fork_git_history") is False, "v2 unexpectedly requires fork Git history")

    report = _load_json(report_payload, "v2 report")
    trials = _load_jsonl(trials_payload, "v2 trials")
    _require(report.get("schema_version") == 2 and report.get("status") == "complete", "v2 report is incomplete")
    _require(report.get("evidence_version") == "v2-post-formal-exploratory", "v2 report classification changed")
    classification = _mapping(report.get("analysis_classification"), "v2 classification")
    for field in ("held_out", "pre_registered", "confirmatory", "model_rerun", "production_generalization_claimed"):
        _require(classification.get(field) is False, f"v2 classification changed: {field}")
    _require(classification.get("v1_remains_sole_formal") is True, "v2 report displaced v1")
    _require(classification.get("reused_v1_raw_outputs") == 96, "v2 replay denominator changed")
    _require("primary_inference" not in report and "quality_gate" not in report, "v2 contains a formal inference section")
    methods = _mapping(report.get("methods"), "v2 methods")
    _require(methods.get("confirmatory_p_values") == "omitted", "v2 reports confirmatory p-values")
    gate = _mapping(report.get("exploratory_gate"), "v2 exploratory gate")
    _require(gate.get("result") == "fail" and gate.get("affects_process_exit") is True, "v2 exploratory gate changed")
    result = _mapping(manifest.get("result"), "v2 publication result")
    _require(result.get("exploratory_gate") == gate.get("result"), "v2 result/gate mismatch")
    _require(result.get("model_rerun") is False and result.get("confirmatory") is False, "v2 result classification changed")
    output_integrity = _mapping(report.get("output_integrity"), "v2 output integrity")
    _require(output_integrity.get("trials_sha256") == _sha256(trials_payload), "v2 report/trials binding mismatch")
    _require(output_integrity.get("v2_policy_plan_sha256") == _sha256(policy_plan_payload), "v2 report/policy-plan binding mismatch")
    _require(
        _mapping(output_integrity.get("v2_code_freeze"), "v2 report freeze").get("bundle_sha256")
        == freeze.get("bundle_sha256"),
        "v2 report/code-freeze bundle mismatch",
    )
    integrity = _mapping(report.get("input_integrity"), "v2 input integrity")
    expected_integrity = {
        "dataset_sha256": _mapping(inputs["dataset"], "dataset binding").get("sha256"),
        "evaluation_sha256": _sha256(_canonical_lines(evaluation)),
        "protocol_sha256": _mapping(inputs["protocol"], "protocol binding").get("sha256"),
        "freeze_manifest_sha256": _mapping(inputs["freeze"], "freeze binding").get("sha256"),
        "evidence_sha256": _mapping(inputs["v1_evidence"], "v1 evidence binding").get("sha256"),
        "model_run_metadata_sha256": _mapping(inputs["v1_metadata"], "v1 metadata binding").get("sha256"),
        "v1_manifest_sha256": _mapping(inputs["v1_manifest"], "v1 manifest binding").get("sha256"),
        "v2_policy_plan_sha256": _mapping(inputs["v2_policy_plan"], "v2 plan binding").get("sha256"),
        "v2_code_freeze_manifest_sha256": _mapping(inputs["v2_code_freeze"], "v2 freeze binding").get("sha256"),
    }
    for field, expected in expected_integrity.items():
        _require(integrity.get(field) == expected, f"v2 report binding mismatch: {field}")
    _require(integrity.get("freeze_verified") is True, "v2 report did not verify the data freeze")
    _require(integrity.get("evidence_pairs_verified") == 96, "v2 report evidence count changed")
    _require(integrity.get("v1_all_artifact_bindings_verified") is True, "v2 report did not bind all v1 artifacts")
    v1_manifest = _load_json(
        _read_bytes(v1_dir / "manifest.json", "v1 manifest"), "v1 manifest"
    )
    _require(
        integrity.get("v1_artifacts_sha256") == v1_manifest.get("artifacts"),
        "v2 report v1-artifact bindings changed",
    )
    report_code = _mapping(integrity.get("code_binding"), "v2 report code binding")
    for field in (
        "current_evaluator_matches_v1",
        "current_runner_matches_v1",
        "current_policy_differs_from_v1",
        "v1_cross_bindings_verified",
        "v1_archived_source_bindings_verified",
    ):
        _require(report_code.get(field) is True, f"v2 report code verification failed: {field}")
    expected_code_hashes = {
        "current_grounding_policy_sha256": executed["clarify_commit.py"],
        "current_evaluator_sha256": executed["evaluate.py"],
        "current_replay_policy_sha256": executed["replay_policy.py"],
        "v1_grounding_policy_sha256": _mapping(v1_manifest.get("code"), "v1 code")["clarify_commit.py"],
        "v1_evaluator_sha256": _mapping(v1_manifest.get("code"), "v1 code")["evaluate.py"],
        "v1_runner_sha256": _mapping(v1_manifest.get("code"), "v1 code")["run_model.py"],
    }
    for field, expected in expected_code_hashes.items():
        _require(report_code.get(field) == expected, f"v2 report code hash mismatch: {field}")
    _require(
        _mapping(report_code.get("v2_code_freeze"), "v2 report source binding").get("bundle_sha256")
        == freeze.get("bundle_sha256"),
        "v2 report source-bundle hash mismatch",
    )
    _verify_aggregates(report, trials, evaluation)
    return {
        "status": "verified",
        "manifest_sha256": PINNED_V2_MANIFEST_SHA256,
        "trial_records": len(trials),
        "exploratory_gate": gate["result"],
        "model_rerun": False,
    }


def _verify_ha(case_dir: Path) -> dict[str, object]:
    evidence_path = case_dir / "evidence" / "ha_acceptance.json"
    payload = _read_pinned(
        evidence_path, PINNED_HA_ACCEPTANCE_SHA256, "Home Assistant acceptance"
    )
    evidence = _load_json(payload, "Home Assistant acceptance")
    _require(evidence.get("schema_version") == 1 and evidence.get("status") == "passed", "HA acceptance did not pass")
    image = _mapping(evidence.get("image"), "HA image")
    _require(image.get("repository") == "ghcr.io/home-assistant/home-assistant", "HA image repository changed")
    _require(image.get("version") == "2026.8.3", "HA image version changed")
    _require(
        image.get("manifest_digest")
        == "sha256:8e9751cb66d3ba6624f5360a7d31b0c6821f7f5b3fb8ba0d10d58f0f481c540c",
        "HA image digest changed",
    )
    _require(image.get("architecture") == "amd64" and image.get("operating_system") == "linux", "HA image platform changed")
    isolation = _mapping(evidence.get("isolation"), "HA isolation")
    _require(isolation.get("container_count") == 1, "HA container count changed")
    _require(isolation.get("named_volume_count") == 1, "HA volume count changed")
    _require(isolation.get("random_loopback_binding") is True, "HA was not loopback-isolated")
    _require(isolation.get("restart_policy") == "no", "HA restart policy changed")
    home_assistant = _mapping(evidence.get("home_assistant"), "HA result")
    health = _mapping(home_assistant.get("health"), "HA health")
    _require(health.get("authenticated_api_http") == 200, "HA authenticated health check failed")
    _require(health.get("unauthenticated_api_http") == 401, "HA unauthenticated boundary changed")
    auth = _mapping(home_assistant.get("auth"), "HA auth")
    _require(auth.get("issue_http") == 200 and auth.get("revoke_http") == 200, "HA token lifecycle failed")
    _require(auth.get("refresh_after_revoke_http") == 400, "HA revoked credential remained usable")
    phases = _mapping(home_assistant.get("phases"), "HA phases")
    setup = _mapping(phases.get("setup"), "HA setup")
    _require(setup.get("included_in_sut_dispatch_count") is False, "HA setup calls entered SUT count")
    sut = _mapping(phases.get("sut"), "HA SUT")
    _require(sut.get("classification") == "clarify_commit_sut", "HA SUT classification changed")
    cases = _sequence(sut.get("cases"), "HA SUT cases")
    expected_cases = {
        "clarified_light_brightness": ("light", "turn_on"),
        "unique_cover_position": ("cover", "set_cover_position"),
        "unique_climate_temperature": ("climate", "set_temperature"),
    }
    _require(len(cases) == sut.get("sut_dispatch_total") == 3, "HA SUT dispatch count mismatch")
    observed_names: set[str] = set()
    for raw_case in cases:
        case = _mapping(raw_case, "HA SUT case")
        name = case.get("case")
        _require(isinstance(name, str) and name in expected_cases, f"unknown HA SUT case: {name}")
        _require(name not in observed_names, f"duplicate HA SUT case: {name}")
        observed_names.add(name)
        grounding = _mapping(case.get("grounding"), f"HA grounding {name}")
        candidates = _sequence(grounding.get("candidate_ids"), f"HA candidates {name}")
        _require(grounding.get("selected_entity_id") in candidates, f"HA selected target not in candidates: {name}")
        postcondition = _mapping(case.get("postcondition"), f"HA postcondition {name}")
        _require(postcondition.get("status") == "COMMITTED", f"HA case was not committed: {name}")
        _require(postcondition.get("reason") == "committed", f"HA commit reason changed: {name}")
        _require(postcondition.get("matched_prepared_projection") is True, f"HA projection mismatch: {name}")
        _require(postcondition.get("all_registered_entities_exact") is True, f"HA exact-state check failed: {name}")
        replay = _mapping(case.get("replay"), f"HA replay {name}")
        _require(replay.get("accepted") is False and replay.get("dispatched") is False, f"HA replay was accepted: {name}")
        _require(replay.get("reason") == "replayed_nonce" and replay.get("sut_dispatch_delta") == 0, f"HA replay guard failed: {name}")
        shape = _mapping(case.get("service_shape"), f"HA service {name}")
        expected_domain, expected_service = expected_cases[name]
        _require(shape.get("domain") == expected_domain and shape.get("service") == expected_service, f"HA service shape mismatch: {name}")
    _require(observed_names == set(expected_cases), "HA SUT case set changed")
    teardown = _mapping(phases.get("teardown"), "HA teardown")
    _require(teardown.get("refresh_revoke_http") == 200, "HA teardown revoke failed")
    _require(teardown.get("refresh_after_revoke_http") == 400, "HA teardown credential remained usable")

    code_freeze = _load_json(
        _read_pinned(
            case_dir / "evidence" / "v2" / "code_freeze.json",
            PINNED_V2_CODE_FREEZE_SHA256,
            "v2 code freeze",
        ),
        "v2 code freeze",
    )
    ha_binding = _mapping(
        _mapping(code_freeze.get("files"), "v2 code-freeze files").get("ha_acceptance.py"),
        "HA source binding",
    )
    _verify_binding(
        _read_bytes(case_dir / "ha_acceptance.py", "HA acceptance source"),
        ha_binding,
        "HA acceptance source",
    )
    return {
        "status": "verified",
        "artifact_sha256": PINNED_HA_ACCEPTANCE_SHA256,
        "image_version": image["version"],
        "sut_cases": len(cases),
    }


def _verify_v3(case_dir: Path) -> dict[str, object]:
    """Verify the post-v2 hardening record without reclassifying it as formal."""

    v3_dir = case_dir / "evidence" / "v3"
    plan_payload = _read_pinned(
        v3_dir / "policy_plan.json",
        PINNED_V3_POLICY_PLAN_SHA256,
        "v3 policy plan",
    )
    validation_payload = _read_pinned(
        v3_dir / "validation.json",
        PINNED_V3_VALIDATION_SHA256,
        "v3 validation",
    )
    plan = _load_json(plan_payload, "v3 policy plan")
    validation = _load_json(validation_payload, "v3 validation")
    _require(plan.get("schema_version") == 1, "v3 policy-plan schema changed")
    _require(
        plan.get("evidence_version") == "v3-post-v2-hardening",
        "v3 policy-plan evidence version changed",
    )
    _require(
        plan.get("status") == "frozen-before-implementation",
        "v3 policy plan is not pre-implementation",
    )
    plan_classification = _mapping(
        plan.get("analysis_classification"),
        "v3 policy-plan classification",
    )
    for field in ("confirmatory", "held_out", "model_rerun", "raw_output_replay"):
        _require(
            plan_classification.get(field) is False,
            f"v3 plan classification changed: {field}",
        )
    _require(
        plan_classification.get("v1_remains_sole_formal") is True,
        "v3 plan displaced v1",
    )
    _require(
        plan_classification.get("v2_record_remains_immutable") is True,
        "v3 plan displaced v2",
    )
    invariant_ids = {
        str(_mapping(item, "v3 invariant").get("id"))
        for item in _sequence(plan.get("invariants"), "v3 invariants")
    }
    _require(
        invariant_ids
        == {
            "punctuation-insensitive-negative-selector",
            "negative-clause-anaphora",
            "no-negated-target-dispatch",
            "preserve-positive-contrast",
        },
        "v3 invariant set changed",
    )

    _require(validation.get("schema_version") == 1, "v3 validation schema changed")
    _require(validation.get("status") == "validated", "v3 validation did not pass")
    _require(
        validation.get("evidence_version") == "v3-post-v2-hardening",
        "v3 validation evidence version changed",
    )
    classification = _mapping(
        validation.get("analysis_classification"),
        "v3 validation classification",
    )
    for field in ("confirmatory", "held_out", "model_rerun", "official_v2_replay"):
        _require(
            classification.get(field) is False,
            f"v3 validation classification changed: {field}",
        )
    _require(
        classification.get("v1_remains_sole_formal") is True,
        "v3 validation displaced v1",
    )
    _require(
        classification.get("v2_record_remains_immutable") is True,
        "v3 validation displaced v2",
    )

    commit = _mapping(validation.get("hardening_commit"), "v3 hardening commit")
    _require(
        commit.get("sha") == PINNED_V3_HARDENING_COMMIT,
        "v3 hardening commit changed",
    )
    _require(
        commit.get("subject") == "fix: harden negated selector scope",
        "v3 commit subject changed",
    )
    _require(
        commit.get("signed_off_by")
        == "MittaPei <315415437+MittaPei@users.noreply.github.com>",
        "v3 sign-off changed",
    )

    immutable = _mapping(
        validation.get("immutable_evidence"),
        "v3 immutable evidence",
    )
    _require(
        immutable.get("v1_manifest_sha256") == PINNED_V1_MANIFEST_SHA256,
        "v3/v1 binding changed",
    )
    _require(
        immutable.get("v2_manifest_sha256") == PINNED_V2_MANIFEST_SHA256,
        "v3/v2 binding changed",
    )
    _require(
        immutable.get("home_assistant_acceptance_sha256")
        == PINNED_HA_ACCEPTANCE_SHA256,
        "v3/HA binding changed",
    )

    bindings = _mapping(
        validation.get("source_bindings"),
        "v3 source bindings",
    )
    expected_sources = {
        "clarify_commit.py": case_dir / "clarify_commit.py",
        "tests/test_clarify_commit.py": case_dir / "tests" / "test_clarify_commit.py",
        "evidence/v3/policy_plan.json": v3_dir / "policy_plan.json",
    }
    _require(
        set(bindings) == set(expected_sources),
        "v3 source-binding set changed",
    )
    for relative, path in expected_sources.items():
        payload = (
            plan_payload
            if relative == "evidence/v3/policy_plan.json"
            else _read_bytes(path, f"v3 source {relative}")
        )
        _verify_binding(payload, bindings[relative], f"v3 source {relative}")

    results = _mapping(
        validation.get("validation_results"),
        "v3 validation results",
    )
    expected_results: dict[str, tuple[str, int | None]] = {
        "policy_suite": (
            "python -m unittest cases/domux-clarify-commit/tests/test_clarify_commit.py",
            106,
        ),
        "case_full_suite": (
            "python -m unittest discover -q -s cases/domux-clarify-commit/tests",
            184,
        ),
        "ruff": ("ruff check cases/domux-clarify-commit", None),
        "python_compile": (
            "python -m py_compile cases/domux-clarify-commit/*.py cases/domux-clarify-commit/tests/*.py",
            None,
        ),
        "diff_check": ("git diff --check", None),
    }
    _require(
        set(results) == set(expected_results),
        "v3 validation-result set changed",
    )
    for name, (command, passed) in expected_results.items():
        result = _mapping(results[name], f"v3 validation result {name}")
        _require(
            result.get("command") == command,
            f"v3 validation command changed: {name}",
        )
        _require(
            result.get("result") == "passed",
            f"v3 validation result failed: {name}",
        )
        if passed is not None:
            _require(
                result.get("passed") == passed,
                f"v3 validation count changed: {name}",
            )

    reproductions = _mapping(
        validation.get("reproduction_results"),
        "v3 reproduction results",
    )
    _require(set(reproductions) == {"v1", "v2"}, "v3 reproduction set changed")
    v1_reproduction = _mapping(reproductions["v1"], "v3 v1 reproduction")
    _require(
        v1_reproduction.get("command")
        == "python cases/domux-clarify-commit/reproduce_v1.py",
        "v3 v1 reproduction command changed",
    )
    _require(
        v1_reproduction.get("expected_evaluator_exit_code") == 1
        and v1_reproduction.get("quality_gate") == "fail"
        and v1_reproduction.get("model_rerun") is False
        and v1_reproduction.get("result") == "byte_identical",
        "v3 v1 reproduction classification changed",
    )
    _require(
        _mapping(v1_reproduction.get("files"), "v3 v1 reproduction files")
        == {
            "report.json": "edea57b50e0c9ea789ca97252ada87e7064d87b8ac739c0f019519441ed2be97",
            "trials.jsonl": "c7f0c97943bd49f4c21306eadeb38a69de8b063e0d821d21ee966c03ee287171",
        },
        "v3 v1 reproduction hashes changed",
    )
    v2_reproduction = _mapping(reproductions["v2"], "v3 v2 reproduction")
    _require(
        v2_reproduction.get("command")
        == "python cases/domux-clarify-commit/reproduce_v2.py",
        "v3 v2 reproduction command changed",
    )
    _require(
        v2_reproduction.get("expected_replay_exit_code") == 1
        and v2_reproduction.get("exploratory_gate") == "fail"
        and v2_reproduction.get("model_rerun") is False
        and v2_reproduction.get("result") == "byte_identical",
        "v3 v2 reproduction classification changed",
    )
    _require(
        _mapping(v2_reproduction.get("files"), "v3 v2 reproduction files")
        == {
            "manifest.json": PINNED_V2_MANIFEST_SHA256,
            "report.json": "b9df72353b2d20125ec75279e686e6316569dcc16b7b3db0666845a04452ebe9",
            "trials.jsonl": "e7a02a537ccf88afd101172ca2bafd8d3acc077b2030dc03b8ca7b5b1bf2ef5e",
        },
        "v3 v2 reproduction hashes changed",
    )

    review = _mapping(
        validation.get("independent_review"),
        "v3 independent review",
    )
    _require(review.get("review_passes") == 2, "v3 independent-review count changed")
    _require(
        review.get("method")
        == "two isolated AI-assisted read-only code-review passes followed by main-agent verification",
        "v3 independent-review method changed",
    )
    _require(review.get("blockers") == 0, "v3 independent review contains a blocker")
    _require(
        review.get("major_findings") == 0,
        "v3 independent review contains a major finding",
    )
    matrices = _mapping(
        review.get("validated_e2e_matrices"),
        "v3 review matrices",
    )
    _require(
        matrices
        == {
            "generic_alias_negative_and_positive_pairs": 40,
            "negative_scope_and_restart_boundaries": 28,
            "room_domain_overlap_and_repair": 6,
        },
        "v3 independent-review matrix changed",
    )
    _require(
        len(_sequence(validation.get("non_claims"), "v3 non-claims")) == 4,
        "v3 non-claim set changed",
    )
    return {
        "status": "verified",
        "validation_sha256": PINNED_V3_VALIDATION_SHA256,
        "hardening_commit": PINNED_V3_HARDENING_COMMIT,
        "policy_tests": 106,
        "full_tests": 184,
        "frozen_reproductions": 2,
        "model_rerun": False,
        "official_v2_replay": False,
    }


def verify_all(case_dir: Path = CASE_DIR) -> dict[str, object]:
    """Verify all immutable evidence and return a deterministic summary."""

    case_dir = Path(case_dir)
    _, evaluation = _verify_frozen_data(case_dir)
    v1 = _verify_v1(case_dir, evaluation)
    v2 = _verify_v2(case_dir, evaluation)
    ha = _verify_ha(case_dir)
    v3 = _verify_v3(case_dir)
    return {
        "status": "verified",
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "home_assistant": ha,
    }


def compare_replay_directory(
    compare_dir: Path,
    *,
    official_dir: Path = CASE_DIR / "evidence" / "v2",
) -> dict[str, object]:
    """Require a replay directory to be byte-identical to the official v2 set."""

    compare_dir = Path(compare_dir)
    official_dir = Path(official_dir)
    digests: dict[str, str] = {}
    for name in REPLAY_OUTPUT_NAMES:
        official = _read_bytes(official_dir / name, f"official v2 {name}")
        candidate = _read_bytes(compare_dir / name, f"comparison v2 {name}")
        _require(candidate == official, f"comparison artifact differs byte-for-byte: {name}")
        digests[name] = _sha256(official)
    return {"status": "byte_identical", "files": digests}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        type=Path,
        help="directory containing manifest.json, report.json, and trials.jsonl to compare byte-for-byte",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_all()
        if args.compare is not None:
            result["comparison"] = compare_replay_directory(args.compare)
    except (VerificationError, OSError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
