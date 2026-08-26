#!/usr/bin/env python3
"""Offline verifier for the immutable Domux case evidence.

The verifier intentionally uses only the Python standard library.  It checks
the frozen v1 model run, the post-formal v2 replay publication, and the pinned
Home Assistant acceptance record.  It also preserves the historical v3 record
and binds the final v4 submission-readiness closure without executing the
model, Docker, Home Assistant, or any network operation.  The post-formal
first-match diagnostic is deterministically rebuilt with its pinned v1 code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import statistics
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
PINNED_V1_FIRST_MATCH_DIAGNOSTIC_SHA256 = (
    "caeb2529c650441dcc56b9f9a67d6f05a984460bb3004d64ab9912e789acbf33"
)
PINNED_V4_HA_ACCEPTANCE_SHA256 = (
    "4af8c185ff14e8c6d89f4b942ead7bd1f06685c83a8648d0e1d4fed3ad9e7cc3"
)
PINNED_HA_ACCEPTANCE_SHA256 = (
    "7bb74408f84765f22e24dcb3863ba150d9f5797ec369792adedb93e7c8f10346"
)
PINNED_V1_DOMUX_RAW_SHA256 = (
    "c0561bc72042dc7415d322fea90649866355dc44d2547f246d87cd87d367e966"
)
PINNED_SCENARIO_EVIDENCE_SHA256 = (
    "0e27842c62d9cd4e4b1467b43e3ebcd346c79c0125c4f40cce97d363c821a0a0"
)
HA_REGISTRY_PROFILE = "semantic_target_mapping_subset_not_full_scenario_inventory"
PINNED_V3_HA_ACCEPTANCE_SHA256 = (
    "fc3132b74978e3bb73954a681800cc13b398dd57267a0be953c83fd23e40d1e7"
)
PINNED_V3_POLICY_PLAN_SHA256 = (
    "025a5ed08416bb3b41f6fcd5ecb90e769cc47ead46eb9caf863a6f3b467895d4"
)
PINNED_V3_VALIDATION_SHA256 = (
    "da2d5d3fef495de1196e95ac456bce7733068de40fbf2ec971eee4c3266184f0"
)
PINNED_V3_HARDENING_COMMIT = "482b94eea78ac198f2abbfac5f2f16da02fb7b9e"
PINNED_V4_VALIDATION_SHA256 = (
    "3966044c0e95a352d839bfa2639f45edd939ed1d9bd3b2e7c715d23beef49455"
)
PINNED_V4_IMPLEMENTATION_COMMIT = "80b2b6c9f65f7ba566c4f308cbbc5692636ca26b"

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
    "ha_acceptance.py",
    "tests/test_clarify_commit.py",
    "tests/test_ha_acceptance.py",
)
V4_IMPLEMENTATION_SOURCE_FILES = (
    "clarify_commit.py",
    "ha_acceptance.py",
    "reproduce_v2.py",
    "requirements.txt",
    "tests/test_clarify_commit.py",
    "tests/test_ha_acceptance.py",
    "tests/test_reproduce_v2.py",
)
V4_VALIDATION_HARNESS_FILES = ("tests/test_verify_artifacts.py",)
V4_PRESENTATION_FILES = ("preview.png", "preview.svg")
V4_SOURCE_FILES = (
    V4_IMPLEMENTATION_SOURCE_FILES
    + V4_PRESENTATION_FILES
    + V4_VALIDATION_HARNESS_FILES
)
V4_ARCHIVED_SOURCE_PATHS = {
    "ha_acceptance.py": "evidence/v4/code/ha_acceptance.py",
    "preview.png": "evidence/v4/presentation/preview.png",
    "preview.svg": "evidence/v4/presentation/preview.svg",
    "tests/test_ha_acceptance.py": (
        "evidence/v4/code/tests/test_ha_acceptance.py"
    ),
    "tests/test_verify_artifacts.py": (
        "evidence/v4/code/tests/test_verify_artifacts.py"
    ),
}
REPLAY_OUTPUT_NAMES = ("trials.jsonl", "report.json", "manifest.json")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
WILSON_95_Z = 1.959963984540054


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


def _finite_number(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not a finite number",
    )
    return float(value)


def _require_close(
    observed: object,
    expected: float,
    label: str,
    *,
    absolute_tolerance: float = 1e-14,
) -> None:
    numeric = _finite_number(observed, label)
    _require(
        math.isclose(
            numeric,
            expected,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ),
        f"{label} mismatch",
    )


def _wilson_interval(successes: int, denominator: int) -> tuple[float, float]:
    _require(
        denominator > 0 and 0 <= successes <= denominator,
        "Wilson inputs are invalid",
    )
    proportion = successes / denominator
    z2 = WILSON_95_Z * WILSON_95_Z
    scale = 1.0 + z2 / denominator
    center = (proportion + z2 / (2.0 * denominator)) / scale
    half = WILSON_95_Z / scale * math.sqrt(
        proportion * (1.0 - proportion) / denominator
        + z2 / (4.0 * denominator * denominator)
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == denominator else min(1.0, center + half)
    return lower, upper


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
    observed_successes = metric.get("successes")
    observed_denominator = metric.get("denominator")
    _require(
        isinstance(observed_successes, int)
        and not isinstance(observed_successes, bool),
        f"{label} successes is not an integer",
    )
    _require(
        isinstance(observed_denominator, int)
        and not isinstance(observed_denominator, bool),
        f"{label} denominator is not an integer",
    )
    _require(observed_successes == successes, f"{label} successes mismatch")
    _require(observed_denominator == denominator, f"{label} denominator mismatch")
    expected_rate = successes / denominator
    _require_close(metric.get("rate"), expected_rate, f"{label} rate")
    wilson = _mapping(metric.get("wilson_95"), f"{label} Wilson 95% interval")
    lower, upper = _wilson_interval(successes, denominator)
    _require_close(
        wilson.get("lower"),
        lower,
        f"{label} Wilson lower bound",
    )
    _require_close(
        wilson.get("upper"),
        upper,
        f"{label} Wilson upper bound",
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


def _verify_language_latency(
    report: Mapping[str, object],
    language: Mapping[tuple[str, str], Mapping[str, object]],
    base_ids: Sequence[str],
) -> None:
    methods = _mapping(report.get("methods"), "methods")
    _require(
        methods.get("binary_intervals")
        == "Wilson two-sided 95%, z=1.959963984540054",
        "binary interval method changed",
    )

    paired: list[float] = []
    missing = 0
    for base_id in base_ids:
        values = [language[(base_id, variant)].get("latency_ms") for variant in VARIANTS]
        numeric: list[float | None] = []
        for variant, value in zip(VARIANTS, values):
            if value is None:
                numeric.append(None)
                continue
            observed = _finite_number(value, f"language latency {base_id}/{variant}")
            _require(
                observed >= 0.0,
                f"language latency is negative: {base_id}/{variant}",
            )
            numeric.append(observed)
        if any(value is None for value in values):
            missing += 1
            continue
        _require(
            all(value is not None for value in numeric),
            f"language latency pair is incomplete: {base_id}",
        )
        paired.append(
            float(statistics.median(value for value in numeric if value is not None))
        )

    latency = _mapping(
        _mapping(
            _mapping(report.get("metrics"), "metrics").get("language"),
            "language metrics",
        ).get("latency"),
        "language latency",
    )
    _require(
        latency.get("unit") == "within-base median of clear and ambiguous latency",
        "language latency unit changed",
    )
    _require(
        latency.get("formal_base_denominator") == len(base_ids) == FORMAL_BASE_COUNT,
        "language latency denominator mismatch",
    )
    _require(latency.get("complete_pairs") == len(paired), "language latency complete-pair count mismatch")
    _require(latency.get("missing_pairs") == missing, "language latency missing-pair count mismatch")
    _require(len(paired) + missing == len(base_ids), "language latency pair accounting mismatch")

    if not paired:
        _require(latency.get("median_ms") is None, "language latency median mismatch")
        _require(
            latency.get("p95_ms_nearest_rank") is None,
            "language latency p95 mismatch",
        )
        return

    ordered = sorted(paired)
    expected_median = float(statistics.median(paired))
    expected_p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    _require_close(
        latency.get("median_ms"),
        expected_median,
        "language latency median",
        absolute_tolerance=1e-12,
    )
    _require_close(
        latency.get("p95_ms_nearest_rank"),
        expected_p95,
        "language latency p95 nearest-rank",
        absolute_tolerance=1e-12,
    )


def _exact_mcnemar(
    comparison_id: str,
    arm_a: str,
    arm_b: str,
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
) -> dict[str, object]:
    _require(
        len(outcomes_a) == len(outcomes_b) and bool(outcomes_a),
        f"McNemar outcome vectors are invalid: {comparison_id}",
    )
    both = sum(a and b for a, b in zip(outcomes_a, outcomes_b))
    a_only = sum(a and not b for a, b in zip(outcomes_a, outcomes_b))
    b_only = sum(not a and b for a, b in zip(outcomes_a, outcomes_b))
    neither = len(outcomes_a) - both - a_only - b_only
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(a_only, b_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "comparison_id": comparison_id,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "paired_bases": len(outcomes_a),
        "both_success": both,
        "a_only_success": a_only,
        "b_only_success": b_only,
        "neither_success": neither,
        "discordant_pairs": discordant,
        "risk_difference_b_minus_a": (b_only - a_only) / len(outcomes_a),
        "exact_two_sided_p": p_value,
    }


def _with_holm_adjustment(
    comparisons: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(
        enumerate(comparisons),
        key=lambda item: (
            float(item[1]["exact_two_sided_p"]),
            str(item[1]["comparison_id"]),
        ),
    )
    adjusted = [1.0] * len(comparisons)
    running = 0.0
    for rank, (original_index, comparison) in enumerate(ordered):
        candidate = min(
            1.0,
            (len(comparisons) - rank) * float(comparison["exact_two_sided_p"]),
        )
        running = max(running, candidate)
        adjusted[original_index] = running
    result: list[dict[str, object]] = []
    for index, comparison in enumerate(comparisons):
        updated = dict(comparison)
        updated["holm_adjusted_p"] = adjusted[index]
        updated["reject_at_0_05"] = adjusted[index] <= 0.05
        result.append(updated)
    return result


def _verify_declared_primary_inference(
    report: Mapping[str, object],
    execution: Mapping[tuple[str, str, str], Mapping[str, object]],
    base_ids: Sequence[str],
) -> None:
    inference = _mapping(report.get("primary_inference"), "primary inference")
    _require(
        inference.get("method") == "two-sided exact McNemar on paired base outcomes",
        "primary-inference McNemar method changed",
    )
    _require(
        inference.get("multiplicity")
        == "Holm correction across two pre-registered comparisons",
        "primary-inference multiplicity method changed",
    )

    b0_clean = [
        execution[(base_id, ARMS[0], "clean")].get("exact_delta_success") is True
        for base_id in base_ids
    ]
    b1_clean = [
        execution[(base_id, ARMS[1], "clean")].get("exact_delta_success") is True
        for base_id in base_ids
    ]
    universal = {
        arm: [
            all(
                execution[(base_id, arm, mutation)].get("oracle_pass") is True
                for mutation in UNIVERSAL_GUARD_MUTATIONS
            )
            for base_id in base_ids
        ]
        for arm in ARMS[1:]
    }
    expected = _with_holm_adjustment(
        [
            _exact_mcnemar(
                "B1_vs_B0_ambiguous_clean_exact_delta",
                ARMS[0],
                ARMS[1],
                b0_clean,
                b1_clean,
            ),
            _exact_mcnemar(
                "B2_vs_B1_universal_guard",
                ARMS[1],
                ARMS[2],
                universal[ARMS[1]],
                universal[ARMS[2]],
            ),
        ]
    )
    observed = list(_sequence(inference.get("comparisons"), "primary comparisons"))
    _require(len(observed) == len(expected) == 2, "primary comparison count changed")
    integer_fields = (
        "paired_bases",
        "both_success",
        "a_only_success",
        "b_only_success",
        "neither_success",
        "discordant_pairs",
    )
    numeric_fields = (
        "risk_difference_b_minus_a",
        "exact_two_sided_p",
        "holm_adjusted_p",
    )
    for index, expected_comparison in enumerate(expected):
        comparison = _mapping(observed[index], f"primary comparison {index + 1}")
        label = str(expected_comparison["comparison_id"])
        for field in ("comparison_id", "arm_a", "arm_b"):
            _require(
                comparison.get(field) == expected_comparison[field],
                f"{label} {field} mismatch",
            )
        for field in integer_fields:
            observed_integer = comparison.get(field)
            _require(
                isinstance(observed_integer, int)
                and not isinstance(observed_integer, bool),
                f"{label} {field} is not an integer",
            )
            _require(
                observed_integer == expected_comparison[field],
                f"{label} {field} mismatch",
            )
        for field in numeric_fields:
            _require_close(
                comparison.get(field),
                float(expected_comparison[field]),
                f"{label} {field}",
            )
        _require(
            comparison.get("reject_at_0_05")
            is expected_comparison["reject_at_0_05"],
            f"{label} reject_at_0_05 mismatch",
        )


def _verify_aggregates(
    report: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
    evaluation: Sequence[Mapping[str, object]],
    *,
    primary_inference_required: bool | None = None,
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
    _verify_language_latency(report, language, base_ids)

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

    primary_inference_present = "primary_inference" in report
    if primary_inference_required is True:
        _require(primary_inference_present, "v1 primary inference is missing")
    elif primary_inference_required is False:
        _require(
            not primary_inference_present,
            "v2 contains a primary inference section",
        )
    if primary_inference_present:
        _verify_declared_primary_inference(report, execution, base_ids)


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
    _verify_aggregates(
        report,
        trials,
        evaluation,
        primary_inference_required=True,
    )
    return {
        "status": "verified",
        "manifest_sha256": PINNED_V1_MANIFEST_SHA256,
        "raw_probes": len(raw),
        "trial_records": len(trials),
        "formal_headline": True,
    }


def _diagnostic_count_metric(
    value: object,
    successes: int,
    denominator: int,
    label: str,
) -> None:
    metric = _mapping(value, label)
    observed_successes = metric.get("successes")
    observed_denominator = metric.get("denominator")
    _require(
        isinstance(observed_successes, int)
        and not isinstance(observed_successes, bool),
        f"{label} successes is not an integer",
    )
    _require(
        isinstance(observed_denominator, int)
        and not isinstance(observed_denominator, bool),
        f"{label} denominator is not an integer",
    )
    _require(observed_successes == successes, f"{label} successes mismatch")
    _require(observed_denominator == denominator, f"{label} denominator mismatch")
    _require_close(metric.get("rate"), successes / denominator, f"{label} rate")


def _rebuild_v1_first_match_diagnostic(case_dir: Path) -> bytes:
    source_path = case_dir / "diagnose_first_match_v1.py"
    module_name = (
        "_domux_verify_first_match_"
        + _sha256(str(source_path.resolve()).encode("utf-8"))[:16]
    )
    policy_path = case_dir / "evidence" / "v1" / "code" / "clarify_commit.py"
    policy_module_name = (
        "_domux_v1_first_match_"
        + _sha256(str(policy_path.resolve()).encode("utf-8"))[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    _require(
        spec is not None and spec.loader is not None,
        "cannot load the pinned first-match diagnostic code",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        build = getattr(module, "build_diagnostic", None)
        render = getattr(module, "render_diagnostic", None)
        _require(
            callable(build) and callable(render),
            "first-match diagnostic API changed",
        )
        rebuilt = build(
            dataset_path=case_dir / "data" / "scenarios.jsonl",
            raw_evidence_path=case_dir / "evidence" / "v1" / "domux_raw.jsonl",
            formal_report_path=case_dir / "evidence" / "v1" / "report.json",
            manifest_path=case_dir / "evidence" / "v1" / "manifest.json",
            policy_path=policy_path,
            diagnostic_code_path=source_path,
        )
        rendered = render(rebuilt)
        _require(
            isinstance(rendered, bytes),
            "first-match diagnostic renderer did not return bytes",
        )
        return rendered
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError("first-match diagnostic rebuild failed") from exc
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(policy_module_name, None)


def _verify_v1_first_match_diagnostic(
    case_dir: Path,
    evaluation: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Verify the post-formal diagnostic by rebuilding its pinned local replay."""

    payload = _read_pinned(
        case_dir / "evidence" / "diagnostics" / "v1_first_match.json",
        PINNED_V1_FIRST_MATCH_DIAGNOSTIC_SHA256,
        "v1 first-match diagnostic",
    )
    diagnostic = _load_json(payload, "v1 first-match diagnostic")
    _require(diagnostic.get("schema_version") == 1, "diagnostic schema changed")
    _require(
        diagnostic.get("diagnostic_id") == "v1-post-formal-naive-first-match",
        "diagnostic identity changed",
    )
    _require(diagnostic.get("status") == "complete", "diagnostic is incomplete")
    _require(
        diagnostic.get("analysis_class") == "post_formal_diagnostic_only",
        "diagnostic analysis class changed",
    )
    _require(
        diagnostic.get("formal_protocol_changed") is False
        and diagnostic.get("formal_metrics_changed") is False,
        "diagnostic was presented as changing the formal analysis",
    )
    _require(diagnostic.get("model_calls") == 0, "diagnostic model-call count changed")
    _require(
        _mapping(diagnostic.get("generation"), "diagnostic generation")
        == {
            "method": "deterministic replay of hash-bound frozen inputs; no timestamps",
            "command": "python diagnose_first_match_v1.py",
        },
        "diagnostic generation declaration changed",
    )
    _require(
        _mapping(diagnostic.get("scope"), "diagnostic scope")
        == {
            "source": "the 48 ambiguous probes in immutable v1 raw evidence",
            "independent_unit": "base_id",
            "base_count": FORMAL_BASE_COUNT,
            "variant": "ambiguous",
            "mutations": "none; clean initial state only",
        },
        "diagnostic scope changed",
    )
    _require(
        _mapping(diagnostic.get("counterfactual_policy"), "counterfactual policy")
        == {
            "label": "D0_post_formal_naive_first_match",
            "parse": "parse every seven-slot line with the frozen v1 parser",
            "matching": "match only Domux output slots with frozen v1 registry semantics",
            "selection": "select the first match in each scenario's pre-frozen inventory order",
            "multi_action": "process parsed lines sequentially against evolving state",
            "clarification": False,
            "prepared_action": False,
            "state_binding": False,
            "replay_or_expiry_guards": False,
        },
        "counterfactual policy declaration changed",
    )
    expected_limitations = [
        "This arm was specified after formal v1 inspection and is diagnostic, not pre-registered evidence.",
        "Inventory order is deterministic and pre-frozen but semantically arbitrary; first-match outcomes are order-dependent.",
        "The fixed balanced conformance suite does not estimate production prevalence or risk.",
        "The diagnostic uses the frozen in-memory adapter, not the separate real Home Assistant acceptance path.",
        "The result does not replace, rename, or enter any formal B0/B1/B2 metric or significance test.",
    ]
    _require(
        list(_sequence(diagnostic.get("limitations"), "diagnostic limitations"))
        == expected_limitations,
        "diagnostic limitations changed",
    )

    input_paths = {
        "dataset": "data/scenarios.jsonl",
        "raw_evidence": "evidence/v1/domux_raw.jsonl",
        "formal_report": "evidence/v1/report.json",
        "v1_manifest": "evidence/v1/manifest.json",
        "v1_policy": "evidence/v1/code/clarify_commit.py",
        "diagnostic_code": "diagnose_first_match_v1.py",
    }
    bindings = _mapping(diagnostic.get("input_bindings"), "diagnostic input bindings")
    _require(set(bindings) == set(input_paths), "diagnostic input-binding set changed")
    for label, relative in input_paths.items():
        binding = _mapping(bindings[label], f"diagnostic {label} binding")
        _require(binding.get("path") == relative, f"diagnostic {label} path changed")
        expected_hash = _require_digest(
            binding.get("sha256"), f"diagnostic {label} hash"
        )
        source = _read_bytes(case_dir / relative, f"diagnostic {label} input")
        _require(
            _sha256(source) == expected_hash,
            f"diagnostic {label} input hash mismatch",
        )

    eval_by_id = {str(row["base_id"]): row for row in evaluation}
    raw_rows = _load_jsonl(
        _read_bytes(
            case_dir / "evidence" / "v1" / "domux_raw.jsonl",
            "diagnostic v1 raw evidence",
        ),
        "diagnostic v1 raw evidence",
    )
    raw_by_id = {
        str(row.get("base_id")): row
        for row in raw_rows
        if row.get("variant") == "ambiguous"
    }
    _require(
        len(raw_by_id) == len(evaluation) == FORMAL_BASE_COUNT
        and set(raw_by_id) == set(eval_by_id),
        "diagnostic ambiguous evidence set changed",
    )
    trials = list(_sequence(diagnostic.get("trials"), "diagnostic trials"))
    _require(len(trials) == FORMAL_BASE_COUNT, "diagnostic trial count changed")
    _require(
        [
            _mapping(trial, f"diagnostic trial {index}").get("base_id")
            for index, trial in enumerate(trials, start=1)
        ]
        == [str(row["base_id"]) for row in evaluation],
        "diagnostic trial order or base set changed",
    )

    verified_trials: list[Mapping[str, object]] = []
    for index, trial_value in enumerate(trials, start=1):
        trial = _mapping(trial_value, f"diagnostic trial {index}")
        base_id = str(trial.get("base_id"))
        scenario = eval_by_id[base_id]
        evidence = raw_by_id[base_id]
        _require(trial.get("variant") == "ambiguous", f"diagnostic variant changed: {base_id}")
        _require(trial.get("category") == scenario.get("category"), f"diagnostic category mismatch: {base_id}")
        _require(
            trial.get("command") == scenario.get("ambiguous_command") == evidence.get("command"),
            f"diagnostic command mismatch: {base_id}",
        )
        _require(
            trial.get("expected_target_entity") == scenario.get("expected_target_entity"),
            f"diagnostic target mismatch: {base_id}",
        )
        _require(
            trial.get("query_sha256")
            == evidence.get("query_sha256")
            == _sha256(str(trial.get("command")).encode("utf-8")),
            f"diagnostic query binding mismatch: {base_id}",
        )
        raw_output = trial.get("raw_output")
        _require(isinstance(raw_output, str) and bool(raw_output), f"diagnostic raw output missing: {base_id}")
        _require(
            raw_output == evidence.get("raw_output")
            and trial.get("raw_output_sha256") == evidence.get("raw_output_sha256")
            and trial.get("raw_output_sha256") == _sha256(raw_output.encode("utf-8")),
            f"diagnostic raw-output binding mismatch: {base_id}",
        )

        instructions = list(
            _sequence(trial.get("instructions"), f"diagnostic instructions {base_id}")
        )
        _require(
            trial.get("instruction_count") == len(instructions) >= 1,
            f"diagnostic instruction count mismatch: {base_id}",
        )
        _require(
            [
                _mapping(item, f"diagnostic instruction {base_id}").get("instruction")
                for item in instructions
            ]
            == raw_output.splitlines(),
            f"diagnostic parsed instructions changed: {base_id}",
        )
        expected_calls: list[dict[str, object]] = []
        for instruction_index, instruction_value in enumerate(instructions, start=1):
            instruction = _mapping(
                instruction_value,
                f"diagnostic instruction {base_id}/{instruction_index}",
            )
            _require(
                instruction.get("index") == instruction_index,
                f"diagnostic instruction index mismatch: {base_id}",
            )
            candidates = list(
                _sequence(
                    instruction.get("candidate_ids_in_frozen_inventory_order"),
                    f"diagnostic candidates {base_id}/{instruction_index}",
                )
            )
            _require(
                all(isinstance(candidate, str) and candidate for candidate in candidates)
                and len(set(candidates)) == len(candidates),
                f"diagnostic candidate list malformed: {base_id}/{instruction_index}",
            )
            selected = instruction.get("selected_entity_id")
            _require(
                selected == (candidates[0] if candidates else None),
                f"diagnostic did not select the first candidate: {base_id}/{instruction_index}",
            )
            outcome = instruction.get("outcome")
            if outcome == "not_dispatched_no_candidate":
                _require(not candidates, f"diagnostic no-candidate outcome changed: {base_id}/{instruction_index}")
            elif outcome == "not_dispatched_non_executable":
                _require(
                    bool(candidates)
                    and isinstance(instruction.get("error_type"), str)
                    and isinstance(instruction.get("error"), str),
                    f"diagnostic non-executable outcome malformed: {base_id}/{instruction_index}",
                )
            elif outcome == "dispatched":
                _require(
                    bool(candidates)
                    and instruction.get("postcondition_match") is True
                    and isinstance(instruction.get("domain"), str)
                    and isinstance(instruction.get("service"), str),
                    f"diagnostic dispatch outcome malformed: {base_id}/{instruction_index}",
                )
                expected_calls.append(
                    {
                        "domain": instruction["domain"],
                        "service": instruction["service"],
                        "entity_id": selected,
                    }
                )
            else:
                raise VerificationError(
                    f"diagnostic instruction outcome changed: {base_id}/{instruction_index}"
                )

        calls = list(_sequence(trial.get("sut_calls"), f"diagnostic calls {base_id}"))
        _require(calls == expected_calls, f"diagnostic SUT-call ledger mismatch: {base_id}")
        _require(trial.get("sut_call_count") == len(calls), f"diagnostic SUT-call count mismatch: {base_id}")
        wrong_target = any(
            _mapping(call, f"diagnostic call {base_id}").get("entity_id")
            != scenario.get("expected_target_entity")
            for call in calls
        )
        _require(
            trial.get("wrong_target_transition") is wrong_target,
            f"diagnostic wrong-target classification mismatch: {base_id}",
        )
        exact = trial.get("exact_delta_success")
        _require(isinstance(exact, bool), f"diagnostic exact-delta flag malformed: {base_id}")
        if exact:
            _require(
                len(calls) == 1 and not wrong_target,
                f"diagnostic exact-delta claim is inconsistent: {base_id}",
            )
        expected_classification = (
            "exact_expected_delta"
            if exact
            else "wrong_target_transition"
            if wrong_target
            else "non_exact_without_wrong_target_transition"
        )
        _require(
            trial.get("classification") == expected_classification,
            f"diagnostic classification mismatch: {base_id}",
        )
        verified_trials.append(trial)

    comparison = _mapping(diagnostic.get("comparison"), "diagnostic comparison")
    _require(
        comparison.get("formal_arm") == ARMS[0],
        "diagnostic formal comparison arm changed",
    )
    formal_report = _load_json(
        _read_bytes(
            case_dir / "evidence" / "v1" / "report.json",
            "diagnostic formal report",
        ),
        "diagnostic formal report",
    )
    formal_b0 = _mapping(
        _mapping(
            _mapping(formal_report.get("metrics"), "formal metrics").get("execution"),
            "formal execution metrics",
        ).get(ARMS[0]),
        "formal B0 metrics",
    )
    formal_names = {
        "exact_delta_success": "ambiguous_clean_exact_delta_success",
        "dispatch_coverage": "dispatch_coverage",
        "wrong_target_transition": "wrong_target_transition_rate",
        "safe_abstention": "safe_abstention_rate",
    }
    formal = _mapping(comparison.get("formal_v1"), "diagnostic formal metrics")
    _require(set(formal) == set(formal_names), "diagnostic formal metric set changed")
    for diagnostic_name, report_name in formal_names.items():
        source_metric = _mapping(formal_b0.get(report_name), f"formal {report_name}")
        successes = source_metric.get("successes")
        denominator = source_metric.get("denominator")
        _require(
            isinstance(successes, int)
            and not isinstance(successes, bool)
            and denominator == FORMAL_BASE_COUNT,
            f"formal metric malformed: {report_name}",
        )
        _diagnostic_count_metric(
            formal.get(diagnostic_name),
            successes,
            FORMAL_BASE_COUNT,
            f"diagnostic formal {diagnostic_name}",
        )

    post = _mapping(
        comparison.get("post_formal_first_match"),
        "diagnostic first-match metrics",
    )
    count_expectations = {
        "structurally_parseable_output": sum(
            int(trial["instruction_count"]) >= 1 for trial in verified_trials
        ),
        "bases_with_any_sut_call": sum(
            int(trial["sut_call_count"]) > 0 for trial in verified_trials
        ),
        "formal_equivalent_dispatch_coverage": sum(
            int(trial["sut_call_count"]) == 1 for trial in verified_trials
        ),
        "exact_delta_success": sum(
            trial.get("exact_delta_success") is True for trial in verified_trials
        ),
        "wrong_target_transition": sum(
            trial.get("wrong_target_transition") is True for trial in verified_trials
        ),
        "multiple_sut_calls": sum(
            int(trial["sut_call_count"]) > 1 for trial in verified_trials
        ),
        "order_sensitive_candidate_set": sum(
            any(
                len(
                    _sequence(
                        _mapping(instruction, "diagnostic instruction").get(
                            "candidate_ids_in_frozen_inventory_order"
                        ),
                        "diagnostic candidates",
                    )
                )
                > 1
                for instruction in _sequence(
                    trial.get("instructions"), "diagnostic instructions"
                )
            )
            for trial in verified_trials
        ),
    }
    for name, successes in count_expectations.items():
        _diagnostic_count_metric(
            post.get(name),
            successes,
            FORMAL_BASE_COUNT,
            f"diagnostic {name}",
        )
    _require(
        post.get("total_sut_calls")
        == sum(int(trial["sut_call_count"]) for trial in verified_trials),
        "diagnostic total SUT-call count mismatch",
    )
    instruction_outcomes = Counter(
        str(_mapping(instruction, "diagnostic instruction").get("outcome"))
        for trial in verified_trials
        for instruction in _sequence(trial.get("instructions"), "diagnostic instructions")
    )
    _require(
        post.get("instruction_outcomes") == dict(sorted(instruction_outcomes.items())),
        "diagnostic instruction-outcome aggregate mismatch",
    )
    classifications = Counter(str(trial.get("classification")) for trial in verified_trials)
    _require(
        post.get("classifications") == dict(sorted(classifications.items())),
        "diagnostic classification aggregate mismatch",
    )

    category_expectations: dict[str, dict[str, int]] = {}
    for category in sorted({str(trial.get("category")) for trial in verified_trials}):
        selected = [trial for trial in verified_trials if trial.get("category") == category]
        category_expectations[category] = {
            "bases": len(selected),
            "bases_with_any_sut_call": sum(
                int(trial["sut_call_count"]) > 0 for trial in selected
            ),
            "exact_delta_successes": sum(
                trial.get("exact_delta_success") is True for trial in selected
            ),
            "wrong_target_transitions": sum(
                trial.get("wrong_target_transition") is True for trial in selected
            ),
        }
    _require(
        comparison.get("by_category") == category_expectations,
        "diagnostic category aggregate mismatch",
    )
    deltas = _mapping(
        comparison.get("count_deltas_first_match_minus_formal"),
        "diagnostic count deltas",
    )
    expected_deltas = {
        "formal_equivalent_dispatches": (
            count_expectations["formal_equivalent_dispatch_coverage"]
            - int(_mapping(formal["dispatch_coverage"], "formal dispatch")["successes"])
        ),
        "exact_delta_successes": (
            count_expectations["exact_delta_success"]
            - int(_mapping(formal["exact_delta_success"], "formal exact delta")["successes"])
        ),
        "wrong_target_transitions": (
            count_expectations["wrong_target_transition"]
            - int(_mapping(formal["wrong_target_transition"], "formal wrong target")["successes"])
        ),
    }
    _require(deltas == expected_deltas, "diagnostic count deltas mismatch")
    rebuilt_payload = _rebuild_v1_first_match_diagnostic(case_dir)
    _require(
        rebuilt_payload == payload,
        "v1 first-match diagnostic differs from its deterministic rebuild",
    )
    return {
        "status": "verified",
        "artifact_sha256": PINNED_V1_FIRST_MATCH_DIAGNOSTIC_SHA256,
        "base_count": len(verified_trials),
        "exact_delta_successes": count_expectations["exact_delta_success"],
        "wrong_target_transitions": count_expectations["wrong_target_transition"],
        "model_calls": 0,
        "formal_metrics_changed": False,
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
    _verify_aggregates(
        report,
        trials,
        evaluation,
        primary_inference_required=False,
    )
    return {
        "status": "verified",
        "manifest_sha256": PINNED_V2_MANIFEST_SHA256,
        "trial_records": len(trials),
        "exploratory_gate": gate["result"],
        "model_rerun": False,
    }


def _load_pinned_v1_domux_records(
    case_dir: Path,
) -> dict[tuple[str, str], tuple[int, Mapping[str, object]]]:
    payload = _read_pinned(
        case_dir / "evidence" / "v1" / "domux_raw.jsonl",
        PINNED_V1_DOMUX_RAW_SHA256,
        "v1 Domux raw evidence",
    )
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError("v1 Domux raw evidence is not UTF-8") from exc
    _require(
        len(lines) == FORMAL_BASE_COUNT * len(VARIANTS),
        "v1 Domux raw evidence line count changed",
    )
    indexed: dict[tuple[str, str], tuple[int, Mapping[str, object]]] = {}
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line), f"v1 Domux raw evidence line {line_number} is blank")
        record = _load_json(
            line.encode("utf-8"),
            f"v1 Domux raw evidence line {line_number}",
        )
        base_id = record.get("base_id")
        variant = record.get("variant")
        _require(
            isinstance(base_id, str)
            and bool(base_id)
            and variant in VARIANTS,
            f"v1 Domux raw evidence key is invalid at line {line_number}",
        )
        key = (base_id, str(variant))
        _require(
            key not in indexed,
            f"v1 Domux raw evidence contains duplicate key: {key}",
        )
        command = record.get("command")
        raw_output = record.get("raw_output")
        _require(
            isinstance(command, str) and isinstance(raw_output, str),
            f"v1 Domux record text is invalid: {key}",
        )
        query_sha256 = _require_digest(
            record.get("query_sha256"),
            f"v1 Domux query {key}",
        )
        raw_output_sha256 = _require_digest(
            record.get("raw_output_sha256"),
            f"v1 Domux raw output {key}",
        )
        _require(
            query_sha256 == _sha256(command.encode("utf-8")),
            f"v1 Domux query hash mismatch: {key}",
        )
        _require(
            raw_output_sha256 == _sha256(raw_output.encode("utf-8")),
            f"v1 Domux raw output hash mismatch: {key}",
        )
        _require(record.get("status") == "ok", f"v1 Domux record failed: {key}")
        indexed[key] = (line_number, record)
    return indexed


def _verify_ha_domux_provenance(
    case: Mapping[str, object],
    expected: Mapping[str, object],
    records: Mapping[tuple[str, str], tuple[int, Mapping[str, object]]],
    used_keys: set[tuple[str, str]],
) -> None:
    name = str(case.get("case"))
    provenance = _mapping(case.get("domux_evidence"), f"HA Domux provenance {name}")
    base_id = expected["base_id"]
    variant = expected["variant"]
    _require(isinstance(base_id, str) and isinstance(variant, str), "invalid expected provenance")
    key = (base_id, variant)
    _require(key in records, f"HA Domux provenance key is absent from v1: {name}")
    _require(key not in used_keys, f"HA Domux provenance key is reused: {key}")
    line_number, record = records[key]
    expected_provenance = {
        "artifact": "evidence/v1/domux_raw.jsonl",
        "artifact_sha256": PINNED_V1_DOMUX_RAW_SHA256,
        "base_id": base_id,
        "line_number": expected["line_number"],
        "pair_verified": True,
        "query_sha256": expected["query_sha256"],
        "raw_output_sha256": expected["raw_output_sha256"],
        "validation": "whole_artifact_and_per_field_sha256",
        "variant": variant,
    }
    _require(provenance == expected_provenance, f"HA Domux provenance changed: {name}")
    _require(
        line_number == expected["line_number"],
        f"HA Domux provenance line mismatch: {name}",
    )
    _require(
        record.get("command") == expected["command"]
        and record.get("raw_output") == expected["raw_output"]
        and record.get("query_sha256") == expected["query_sha256"]
        and record.get("raw_output_sha256") == expected["raw_output_sha256"],
        f"HA Domux source pair mismatch: {name}",
    )
    used_keys.add(key)


def _load_pinned_scenario_records(
    case_dir: Path,
) -> dict[str, tuple[int, str, Mapping[str, object]]]:
    payload = _read_pinned(
        case_dir / "data" / "scenarios.jsonl",
        PINNED_SCENARIO_EVIDENCE_SHA256,
        "frozen scenario evidence",
    )
    lines = payload.splitlines()
    _require(len(lines) == 64, "frozen scenario evidence line count changed")
    indexed: dict[str, tuple[int, str, Mapping[str, object]]] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        _require(bool(raw_line), f"frozen scenario line {line_number} is blank")
        record = _load_json(raw_line, f"frozen scenario line {line_number}")
        base_id = record.get("base_id")
        _require(
            isinstance(base_id, str) and bool(base_id),
            f"frozen scenario identity is invalid at line {line_number}",
        )
        _require(
            base_id not in indexed,
            f"frozen scenario contains duplicate base_id: {base_id}",
        )
        indexed[base_id] = (line_number, _sha256(raw_line), record)
    return indexed


def _verify_ha_scenario_provenance(
    case: Mapping[str, object],
    expected: Mapping[str, object],
    scenario_records: Mapping[str, tuple[int, str, Mapping[str, object]]],
    domux_records: Mapping[tuple[str, str], tuple[int, Mapping[str, object]]],
    used_base_ids: set[str],
) -> None:
    name = str(case.get("case"))
    base_id = expected.get("base_id")
    variant = expected.get("variant")
    _require(
        isinstance(base_id, str) and variant in VARIANTS,
        f"HA expected scenario identity is invalid: {name}",
    )
    _require(base_id in scenario_records, f"HA scenario row is missing: {name}")
    _require(base_id not in used_base_ids, f"HA scenario row is reused: {base_id}")
    line_number, row_sha256, row = scenario_records[base_id]
    _require(
        row.get("schema_version") == 1
        and row.get("split") == "eval"
        and row.get("category") == "duplicate_entity"
        and row.get("ambiguity_expected") is True,
        f"HA frozen scenario classification changed: {name}",
    )

    clear_command = row.get("clear_command")
    ambiguous_command = row.get("ambiguous_command")
    clarification_answer = row.get("clarification_answer")
    confirmed = row.get("confirmed_instruction")
    expected_target = row.get("expected_target_entity")
    candidate_ids = row.get("candidate_entity_ids")
    inventory = row.get("inventory")
    _require(
        isinstance(clear_command, str)
        and isinstance(ambiguous_command, str)
        and isinstance(clarification_answer, str)
        and isinstance(confirmed, dict)
        and isinstance(expected_target, str)
        and isinstance(candidate_ids, list)
        and all(isinstance(candidate, str) for candidate in candidate_ids)
        and len(candidate_ids) == len(set(candidate_ids))
        and expected_target in candidate_ids
        and isinstance(inventory, list),
        f"HA frozen scenario content is invalid: {name}",
    )
    confirmed_fields = (
        "action",
        "device",
        "attribute",
        "value",
        "unit",
        "room",
        "floor",
    )
    _require(
        set(confirmed) == set(confirmed_fields)
        and all(isinstance(confirmed.get(field), str) for field in confirmed_fields),
        f"HA confirmed scenario instruction is invalid: {name}",
    )
    target_rows = [
        item
        for item in inventory
        if isinstance(item, dict) and item.get("entity_id") == expected_target
    ]
    _require(
        len(target_rows) == 1,
        f"HA scenario target inventory entry changed: {name}",
    )
    target = target_rows[0]
    _require(
        set(target) == {"aliases", "device", "domain", "entity_id", "floor", "room"}
        and isinstance(target.get("aliases"), list)
        and all(isinstance(alias, str) for alias in target["aliases"])
        and all(
            isinstance(target.get(field), str)
            for field in ("device", "domain", "entity_id", "floor", "room")
        ),
        f"HA scenario target semantics are invalid: {name}",
    )
    target_semantics = {
        "aliases": list(target["aliases"]),
        "device": target["device"],
        "domain": target["domain"],
        "floor": target["floor"],
        "room": target["room"],
    }
    grounding = _mapping(case.get("grounding"), f"HA grounding {name}")
    ha_demo_entity_id = grounding.get("selected_entity_id")
    _require(
        isinstance(ha_demo_entity_id, str),
        f"HA selected demo entity is invalid: {name}",
    )
    ha_demo_semantics = {
        "light.ceiling_lights": {
            "aliases": [],
            "device": "Light",
            "domain": "light",
            "floor": "Ground Floor",
            "room": "Living Room",
        },
        "cover.hall_window": {
            "aliases": [],
            "device": "Curtain",
            "domain": "cover",
            "floor": "First Floor",
            "room": "Hall",
        },
        "climate.hvac": {
            "aliases": [],
            "device": "AC",
            "domain": "climate",
            "floor": "Second Floor",
            "room": "Bedroom",
        },
        "light.bed_light": {
            "aliases": [],
            "device": "Light",
            "domain": "light",
            "floor": "Ground Floor",
            "room": "Study",
        },
    }
    _require(
        ha_demo_semantics.get(ha_demo_entity_id) == target_semantics
        and case.get("domain") == target_semantics["domain"],
        f"HA scenario-to-demo semantic mapping changed: {name}",
    )

    command = ambiguous_command if variant == "ambiguous" else clear_command
    confirmed_pipe = "|".join(str(confirmed[field]) for field in confirmed_fields)
    binding_payload = {
        "ambiguous_command": ambiguous_command,
        "base_id": base_id,
        "candidate_entity_ids": candidate_ids,
        "clarification_answer": clarification_answer,
        "clear_command": clear_command,
        "confirmed_instruction": confirmed,
        "expected_target_entity_id": expected_target,
        "target_inventory_semantics": target_semantics,
    }
    grounding_candidates = _sequence(
        grounding.get("candidate_ids"), f"HA grounding candidates {name}"
    )
    used_for_resolution = variant == "ambiguous"
    provenance = _mapping(
        case.get("scenario_provenance"),
        f"HA scenario provenance {name}",
    )
    expected_provenance = {
        "artifact": "data/scenarios.jsonl",
        "artifact_sha256": PINNED_SCENARIO_EVIDENCE_SHA256,
        "base_id": base_id,
        "binding_sha256": _sha256(canonical_json(binding_payload).encode("utf-8")),
        "clarification_answer": clarification_answer,
        "clarification_answer_sha256": _sha256(
            clarification_answer.encode("utf-8")
        ),
        "confirmed_instruction": confirmed,
        "confirmed_instruction_sha256": _sha256(confirmed_pipe.encode("utf-8")),
        "expected_target_entity_id": expected_target,
        "frozen_candidate_count": len(candidate_ids),
        "ha_matching_candidate_count": len(grounding_candidates),
        "ha_registry_profile": HA_REGISTRY_PROFILE,
        "inventory_limitation": {
            "full_scenario_inventory_reproduced": False,
            "profile": HA_REGISTRY_PROFILE,
        },
        "line_number": line_number,
        "post_clarification_model_call": False,
        "row_sha256": row_sha256,
        "scenario_target_to_ha_demo_entity": {
            "ha_demo_entity_id": ha_demo_entity_id,
            "scenario_target_entity_id": expected_target,
            "semantic_fields_match": True,
        },
        "source": "frozen_synthetic_scenario_gold",
        "target_inventory_semantics": target_semantics,
        "used_for_resolution": used_for_resolution,
        "variant": variant,
        "variant_command_sha256": _sha256(command.encode("utf-8")),
    }
    _require(
        provenance == expected_provenance,
        f"HA scenario provenance changed: {name}",
    )

    domux_key = (base_id, str(variant))
    _require(
        domux_key in domux_records,
        f"HA scenario has no matching Domux record: {name}",
    )
    _, domux_record = domux_records[domux_key]
    domux_provenance = _mapping(
        case.get("domux_evidence"), f"HA Domux provenance {name}"
    )
    _require(
        domux_record.get("command") == command
        and domux_record.get("query_sha256")
        == expected_provenance["variant_command_sha256"]
        == domux_provenance.get("query_sha256")
        and expected_target
        == expected_provenance["scenario_target_to_ha_demo_entity"][
            "scenario_target_entity_id"
        ]
        and ha_demo_entity_id == grounding.get("selected_entity_id")
        and grounding.get("clarification_required") is used_for_resolution,
        f"HA scenario/Domux/target cross-check changed: {name}",
    )
    used_base_ids.add(base_id)


def _verify_ha(case_dir: Path) -> dict[str, object]:
    payload = _read_pinned(
        case_dir / "evidence" / "ha_acceptance.json",
        PINNED_HA_ACCEPTANCE_SHA256,
        "Home Assistant acceptance",
    )
    evidence = _load_json(payload, "Home Assistant acceptance")
    _require(
        set(evidence) == {"home_assistant", "image", "isolation", "schema_version", "status"},
        "HA acceptance top-level fields changed",
    )
    _require(
        evidence.get("schema_version") == 3 and evidence.get("status") == "passed",
        "HA acceptance did not pass",
    )
    image = _mapping(evidence.get("image"), "HA image")
    _require(
        image
        == {
            "architecture": "amd64",
            "docker_healthcheck": False,
            "manifest_digest": (
                "sha256:8e9751cb66d3ba6624f5360a7d31b0c6821f7f5b3fb8ba0d10d58f0f481c540c"
            ),
            "operating_system": "linux",
            "repository": "ghcr.io/home-assistant/home-assistant",
            "version": "2026.8.3",
        },
        "HA image changed",
    )
    _require(
        _mapping(evidence.get("isolation"), "HA isolation")
        == {
            "container_count": 1,
            "cpu_limit": 1.5,
            "memory_limit_bytes": 2147483648,
            "named_volume_count": 1,
            "pids_limit": 512,
            "random_loopback_binding": True,
            "restart_policy": "no",
        },
        "HA isolation changed",
    )
    home_assistant = _mapping(evidence.get("home_assistant"), "HA result")
    _require(
        set(home_assistant) == {"auth", "health", "onboarding", "phases", "readiness"},
        "HA result fields changed",
    )
    _require(
        _mapping(home_assistant.get("health"), "HA health")
        == {
            "authenticated_api_http": 200,
            "message": "API running.",
            "unauthenticated_api_http": 401,
        },
        "HA health boundary changed",
    )
    _require(
        _mapping(home_assistant.get("auth"), "HA auth")
        == {
            "issue_http": 200,
            "refresh_after_revoke_http": 400,
            "revoke_http": 200,
            "token_type": "Bearer",
            "ttl_seconds": 1800,
        },
        "HA token lifecycle changed",
    )
    _require(
        _mapping(home_assistant.get("readiness"), "HA readiness")
        == {"endpoint": "/api/onboarding", "http": 200},
        "HA readiness changed",
    )
    onboarding_steps = {
        "analytics": False,
        "core_config": False,
        "integration": False,
        "user": False,
    }
    _require(
        _mapping(home_assistant.get("onboarding"), "HA onboarding")
        == {
            "final": {name: True for name in onboarding_steps},
            "initial": onboarding_steps,
            "requests": {
                "analytics_http": 200,
                "core_config_http": 200,
                "integration_http": 200,
                "users_http": 200,
            },
        },
        "HA onboarding changed",
    )
    phases = _mapping(home_assistant.get("phases"), "HA phases")
    _require(
        set(phases) == {"service_call_accounting", "setup", "sut", "teardown"},
        "HA phase fields changed",
    )
    expected_setup_dispatches = [
        {"domain": "light", "http": 200, "service": "turn_on"},
        {"domain": "light", "http": 200, "service": "turn_on"},
        {"domain": "cover", "http": 200, "service": "set_cover_position"},
        {"domain": "climate", "http": 200, "service": "set_hvac_mode"},
        {"domain": "climate", "http": 200, "service": "set_temperature"},
    ]
    _require(
        _mapping(phases.get("setup"), "HA setup")
        == {
            "classification": "direct_rest_state_normalization",
            "dispatches": expected_setup_dispatches,
            "included_in_sut_dispatch_count": False,
            "purpose": "setup_only",
        },
        "HA setup changed",
    )
    sut = _mapping(phases.get("sut"), "HA SUT")
    _require(
        set(sut)
        == {
            "adapter",
            "case_count",
            "cases",
            "classification",
            "domux_evidence",
            "external_fault_injection_count",
            "pipeline",
            "rejected_before_dispatch_count",
            "scenario_evidence",
            "successful_transition_count",
            "sut_dispatch_total",
        },
        "HA SUT fields changed",
    )
    expected_cases: dict[str, dict[str, object]] = {
        "recorded_ambiguous_light_off": {
            "after": {
                "brightness": None,
                "color_temp_kelvin": None,
                "entity_id": "light.ceiling_lights",
                "rgb_color": None,
                "state": "off",
            },
            "base_id": "eval-duplicate_entity-01",
            "before": {
                "brightness": 178,
                "color_temp_kelvin": 3000,
                "entity_id": "light.ceiling_lights",
                "rgb_color": [255, 177, 110],
                "state": "on",
            },
            "command": "Turn off the light.",
            "domain": "light",
            "grounding": {
                "candidate_ids": ["light.ceiling_lights", "light.bed_light"],
                "clarification_required": True,
                "resolution": "resolve_clarification_submission",
                "selected_entity_id": "light.ceiling_lights",
            },
            "line_number": 2,
            "query_sha256": (
                "f27717f08a911d7db2dcafcec7dc4fb5363b9cff40840596c17d548101b6fdcf"
            ),
            "raw_output": "turnOff|Light|*|*|*|*|*",
            "raw_output_sha256": (
                "c5fee12f0a6f2de9ca00f1a3d64485625ddcceba18e34a799ddbfe38196dd76e"
            ),
            "service_shape": {
                "data": {"entity_id": "light.ceiling_lights"},
                "domain": "light",
                "service": "turn_off",
            },
            "variant": "ambiguous",
        },
        "recorded_unique_cover_position": {
            "after": {
                "current_position": 20,
                "entity_id": "cover.hall_window",
                "state": "open",
            },
            "base_id": "eval-duplicate_entity-02",
            "before": {
                "current_position": 80,
                "entity_id": "cover.hall_window",
                "state": "open",
            },
            "command": "Set the Curtain in the Hall on the First Floor to 20 percent.",
            "domain": "cover",
            "grounding": {
                "candidate_ids": ["cover.hall_window"],
                "clarification_required": False,
                "resolution": "resolve_unique_request",
                "selected_entity_id": "cover.hall_window",
            },
            "line_number": 3,
            "query_sha256": (
                "cd4494727bc97d234d9c50f345468ab88b3571230ef610b5ef367d307f30e784"
            ),
            "raw_output": "set|Curtain|position|20|Percent|Hall|First Floor",
            "raw_output_sha256": (
                "1435c4aa085aa8a5470b0cafc14629f15ea12f5ea0fdb255fe0f3fcb90a85edf"
            ),
            "service_shape": {
                "data": {"entity_id": "cover.hall_window", "position": 20},
                "domain": "cover",
                "service": "set_cover_position",
            },
            "variant": "clear",
        },
        "recorded_unique_climate_temperature": {
            "after": {
                "entity_id": "climate.hvac",
                "fan_mode": "on_high",
                "state": "cool",
                "temperature": 22,
            },
            "base_id": "eval-duplicate_entity-03",
            "before": {
                "entity_id": "climate.hvac",
                "fan_mode": "on_high",
                "state": "cool",
                "temperature": 24,
            },
            "command": "Set the AC in the Bedroom on the Second Floor to 22 Celsius.",
            "domain": "climate",
            "grounding": {
                "candidate_ids": ["climate.hvac"],
                "clarification_required": False,
                "resolution": "resolve_unique_request",
                "selected_entity_id": "climate.hvac",
            },
            "line_number": 5,
            "query_sha256": (
                "c892af8646f0e3d0e52226ddf5d5c0ac4fed0d977ced9592e693b2abaf88962b"
            ),
            "raw_output": "set|AC|temperature|22|Celsius|Bedroom|Second Floor",
            "raw_output_sha256": (
                "37282efde74972be3bc4bdab6351683ed9be31dcefbff39e1527ce81a0da58a4"
            ),
            "service_shape": {
                "data": {"entity_id": "climate.hvac", "temperature": 22},
                "domain": "climate",
                "service": "set_temperature",
            },
            "variant": "clear",
        },
    }
    cases = _sequence(sut.get("cases"), "HA SUT cases")
    expected_order = [*expected_cases, "recorded_study_light_state_drift_rejected"]
    _require(
        len(cases) == sut.get("case_count") == 4,
        "HA acceptance case count mismatch",
    )
    _require(
        [str(_mapping(case, "HA SUT case").get("case")) for case in cases]
        == expected_order,
        "HA case order or identity changed",
    )
    _require(
        sut.get("adapter") == "HomeAssistantRESTAdapter"
        and sut.get("classification") == "clarify_commit_sut"
        and sut.get("pipeline")
        == [
            "ground_domux_request",
            "resolve_clarification_submission_or_unique",
            "PreparedActionStore.prepare",
            "PreparedActionStore.commit",
            "HomeAssistantRESTAdapter.call_service",
        ]
        and sut.get("successful_transition_count") == 3
        and sut.get("rejected_before_dispatch_count") == 1
        and sut.get("external_fault_injection_count") == 1
        and sut.get("sut_dispatch_total") == 3,
        "HA SUT outcome accounting changed",
    )
    _require(
        _mapping(sut.get("domux_evidence"), "HA SUT Domux evidence")
        == {
            "artifact": "evidence/v1/domux_raw.jsonl",
            "artifact_sha256": PINNED_V1_DOMUX_RAW_SHA256,
            "pair_count": 4,
            "validation": "whole_artifact_and_per_field_sha256",
        },
        "HA SUT Domux evidence summary changed",
    )
    _require(
        _mapping(sut.get("scenario_evidence"), "HA SUT scenario evidence")
        == {
            "artifact": "data/scenarios.jsonl",
            "artifact_sha256": PINNED_SCENARIO_EVIDENCE_SHA256,
            "case_count": 4,
            "ha_registry_profile": HA_REGISTRY_PROFILE,
        },
        "HA SUT scenario evidence summary changed",
    )
    records = _load_pinned_v1_domux_records(case_dir)
    scenario_records = _load_pinned_scenario_records(case_dir)
    used_keys: set[tuple[str, str]] = set()
    used_scenario_base_ids: set[str] = set()
    postcondition = {
        "all_registered_entities_exact": True,
        "matched_prepared_projection": True,
        "reason": "committed",
        "status": "COMMITTED",
    }
    replay = {
        "accepted": False,
        "dispatched": False,
        "reason": "replayed_nonce",
        "sut_dispatch_delta": 0,
    }
    committed_case_fields = {
        "case",
        "controlled_after",
        "controlled_before",
        "domain",
        "domux_evidence",
        "grounding",
        "ha_registry_profile",
        "outcome",
        "postcondition",
        "replay",
        "scenario_provenance",
        "service_shape",
    }
    committed: dict[str, Mapping[str, object]] = {}
    for raw_case in cases[:3]:
        case = _mapping(raw_case, "HA committed case")
        name = str(case.get("case"))
        expected = expected_cases[name]
        _require(
            set(case) == committed_case_fields,
            f"HA committed case fields changed: {name}",
        )
        _require(
            case.get("outcome") == "COMMITTED"
            and case.get("domain") == expected["domain"]
            and case.get("ha_registry_profile") == HA_REGISTRY_PROFILE,
            f"HA committed outcome changed: {name}",
        )
        _require(case.get("grounding") == expected["grounding"], f"HA grounding changed: {name}")
        _require(case.get("service_shape") == expected["service_shape"], f"HA service changed: {name}")
        _require(case.get("controlled_before") == expected["before"], f"HA before state changed: {name}")
        _require(case.get("controlled_after") == expected["after"], f"HA after state changed: {name}")
        _require(case.get("postcondition") == postcondition, f"HA postcondition changed: {name}")
        _require(case.get("replay") == replay, f"HA replay guard changed: {name}")
        _verify_ha_domux_provenance(case, expected, records, used_keys)
        _verify_ha_scenario_provenance(
            case,
            expected,
            scenario_records,
            records,
            used_scenario_base_ids,
        )
        committed[name] = case

    drift = _mapping(cases[3], "HA target-drift case")
    _require(
        set(drift)
        == {
            "binding",
            "case",
            "controlled_after_external_mutation",
            "controlled_before_external_mutation",
            "domain",
            "domux_evidence",
            "external_mutation",
            "grounding",
            "ha_registry_profile",
            "outcome",
            "rejection",
            "scenario_provenance",
            "service_shape",
        },
        "HA target-drift case fields changed",
    )
    drift_provenance = {
        "base_id": "eval-duplicate_entity-04",
        "command": (
            "Set the Light in the Study on the Ground Floor to 35 percent brightness."
        ),
        "line_number": 7,
        "query_sha256": (
            "01411704354a221b89abc4c31c1d577fa7aecce126d9fd3db966c68bcfe27973"
        ),
        "raw_output": "set|Light|brightness|35|Percent|Study|Ground Floor",
        "raw_output_sha256": (
            "6caea436fc1c6256aa56198c93270913e9323f6a0935267c55389eef078e57a4"
        ),
        "variant": "clear",
    }
    _verify_ha_domux_provenance(drift, drift_provenance, records, used_keys)
    _verify_ha_scenario_provenance(
        drift,
        drift_provenance,
        scenario_records,
        records,
        used_scenario_base_ids,
    )
    _require(len(used_keys) == 4, "HA Domux provenance pair count changed")
    _require(
        len(used_scenario_base_ids) == 4,
        "HA scenario provenance case count changed",
    )
    _require(
        drift.get("domain") == "light"
        and drift.get("outcome") == "REJECTED_BEFORE_DISPATCH"
        and drift.get("ha_registry_profile") == HA_REGISTRY_PROFILE
        and drift.get("grounding")
        == {
            "candidate_ids": ["light.bed_light"],
            "clarification_required": False,
            "resolution": "resolve_unique_request",
            "selected_entity_id": "light.bed_light",
        },
        "HA target-drift grounding changed",
    )
    _require(
        drift.get("service_shape")
        == {
            "data": {"brightness_pct": 35, "entity_id": "light.bed_light"},
            "domain": "light",
            "service": "turn_on",
        },
        "HA target-drift prepared service changed",
    )
    before_drift = {
        "brightness": 166,
        "color_temp_kelvin": 3000,
        "entity_id": "light.bed_light",
        "rgb_color": [255, 177, 110],
        "state": "on",
    }
    after_drift = {
        "brightness": 64,
        "color_temp_kelvin": 3000,
        "entity_id": "light.bed_light",
        "rgb_color": [255, 177, 110],
        "state": "on",
    }
    _require(
        drift.get("controlled_before_external_mutation") == before_drift
        and drift.get("controlled_after_external_mutation") == after_drift,
        "HA target-drift state chain changed",
    )
    _require(
        drift.get("external_mutation")
        == {
            "classification": "out_of_band_fault_injection",
            "data": {"brightness_pct": 25, "entity_id": "light.bed_light"},
            "domain": "light",
            "http_status": 200,
            "included_in_sut_dispatch_count": False,
            "observed_path": "/api/states/light.bed_light",
            "request_path": "/api/services/light/turn_on",
            "service": "turn_on",
            "transport": "home_assistant_rest_api",
        },
        "HA target-drift mutation evidence changed",
    )
    binding = _mapping(drift.get("binding"), "HA drift binding")
    prepared_digest = _require_digest(binding.get("prepared_state_digest"), "HA prepared state")
    before_digest = _require_digest(
        binding.get("before_external_mutation_state_digest"),
        "HA before-mutation state",
    )
    after_digest = _require_digest(
        binding.get("after_external_mutation_state_digest"),
        "HA after-mutation state",
    )
    _require(
        binding
        == {
            "after_external_mutation_state_digest": (
                "80e68977cce90ffd209341d5cdd5fb029c287f2cc719802fd12abbc6ec6a7f06"
            ),
            "before_external_mutation_state_digest": (
                "dcb0369d3d033fa784ccf60d4aa5d20d00ef1ddb458817e281687679dca574c6"
            ),
            "changed_after_external_mutation": True,
            "matched_before_external_mutation": True,
            "prepared_state_digest": (
                "dcb0369d3d033fa784ccf60d4aa5d20d00ef1ddb458817e281687679dca574c6"
            ),
        }
        and prepared_digest
        == before_digest
        == "dcb0369d3d033fa784ccf60d4aa5d20d00ef1ddb458817e281687679dca574c6"
        and after_digest
        == "80e68977cce90ffd209341d5cdd5fb029c287f2cc719802fd12abbc6ec6a7f06",
        "HA target-drift state binding changed",
    )
    rejection = {
        "accepted": False,
        "acknowledged": False,
        "dispatched": False,
        "outcome_unknown": False,
        "reason": "state_changed",
        "status": "INVALIDATED",
        "sut_dispatch_delta": 0,
    }
    _require(drift.get("rejection") == rejection, "HA target-drift rejection changed")

    expected_direct_events = [
        {
            "domain": dispatch["domain"],
            "http_status": 200,
            "phase": "setup",
            "request_path": (
                f"/api/services/{dispatch['domain']}/{dispatch['service']}"
            ),
            "service": dispatch["service"],
        }
        for dispatch in expected_setup_dispatches
    ]
    expected_direct_events.append(
        {
            "domain": "light",
            "http_status": 200,
            "phase": "external_fault_injection",
            "request_path": "/api/services/light/turn_on",
            "service": "turn_on",
        }
    )
    accounting = _mapping(
        phases.get("service_call_accounting"),
        "HA service-call accounting",
    )
    _require(
        accounting
        == {
            "direct_rest_events": expected_direct_events,
            "external_fault_injection": 1,
            "setup_direct_rest": 5,
            "sut_dispatches": 3,
            "total": 9,
        }
        and len(expected_direct_events) + sut["sut_dispatch_total"] == accounting["total"],
        "HA service-call accounting changed",
    )
    _require(
        _mapping(phases.get("teardown"), "HA teardown")
        == {
            "classification": "credential_cleanup",
            "refresh_after_revoke_http": 400,
            "refresh_revoke_http": 200,
        },
        "HA teardown changed",
    )
    return {
        "status": "verified",
        "artifact_sha256": PINNED_HA_ACCEPTANCE_SHA256,
        "domux_evidence_pairs": len(used_keys),
        "drift_sut_dispatch_delta": rejection["sut_dispatch_delta"],
        "image_version": image["version"],
        "rejected_before_dispatch": 1,
        "successful_transitions": 3,
        "sut_cases": len(cases),
        "sut_dispatch_total": 3,
        "total_service_calls": accounting["total"],
    }


def _verify_v4_ha_archive(case_dir: Path) -> dict[str, object]:
    """Read the immutable v4 HA record without applying the current contract."""

    payload = _read_pinned(
        case_dir / "evidence" / "v4" / "ha_acceptance.json",
        PINNED_V4_HA_ACCEPTANCE_SHA256,
        "Home Assistant acceptance",
    )
    evidence = _load_json(payload, "v4 Home Assistant acceptance")
    _require(
        evidence.get("schema_version") == 2 and evidence.get("status") == "passed",
        "v4 Home Assistant acceptance did not pass",
    )
    image = _mapping(evidence.get("image"), "v4 HA image")
    _require(
        image.get("repository") == "ghcr.io/home-assistant/home-assistant"
        and image.get("version") == "2026.8.3",
        "v4 HA image changed",
    )
    phases = _mapping(
        _mapping(evidence.get("home_assistant"), "v4 HA result").get("phases"),
        "v4 HA phases",
    )
    setup = _mapping(phases.get("setup"), "v4 HA setup")
    _require(
        len(_sequence(setup.get("dispatches"), "v4 HA setup dispatches")) == 4
        and setup.get("included_in_sut_dispatch_count") is False,
        "v4 HA setup summary changed",
    )
    sut = _mapping(phases.get("sut"), "v4 HA SUT")
    cases = _sequence(sut.get("cases"), "v4 HA SUT cases")
    _require(
        [str(_mapping(case, "v4 HA case").get("case")) for case in cases]
        == [
            "clarified_light_brightness",
            "unique_cover_position",
            "unique_climate_temperature",
            "target_state_drift_rejected",
        ]
        and sut.get("case_count") == len(cases) == 4
        and sut.get("successful_transition_count") == 3
        and sut.get("rejected_before_dispatch_count") == 1
        and sut.get("external_fault_injection_count") == 1
        and sut.get("sut_dispatch_total") == 3,
        "v4 HA SUT summary changed",
    )
    drift = _mapping(cases[3], "v4 HA target-drift case")
    binding = _mapping(drift.get("binding"), "v4 HA drift binding")
    _require(
        drift.get("outcome") == "REJECTED_BEFORE_DISPATCH"
        and binding
        == {
            "after_external_mutation_state_digest": (
                "80e68977cce90ffd209341d5cdd5fb029c287f2cc719802fd12abbc6ec6a7f06"
            ),
            "before_external_mutation_state_digest": (
                "bf343f777d3bbb8d7dd36a3d821f5e3c2f206a0d4f7fcfe54da9b043ec6ce7a4"
            ),
            "changed_after_external_mutation": True,
            "matched_before_external_mutation": True,
            "prepared_state_digest": (
                "bf343f777d3bbb8d7dd36a3d821f5e3c2f206a0d4f7fcfe54da9b043ec6ce7a4"
            ),
        }
        and _mapping(drift.get("rejection"), "v4 HA drift rejection").get(
            "sut_dispatch_delta"
        )
        == 0,
        "v4 HA target-drift summary changed",
    )
    accounting = _mapping(
        phases.get("service_call_accounting"),
        "v4 HA service-call accounting",
    )
    _require(
        accounting.get("setup_direct_rest") == 4
        and accounting.get("sut_dispatches") == 3
        and accounting.get("external_fault_injection") == 1
        and accounting.get("total") == 8,
        "v4 HA service-call summary changed",
    )
    return {
        "artifact_sha256": PINNED_V4_HA_ACCEPTANCE_SHA256,
        "drift_sut_dispatch_delta": 0,
        "rejected_before_dispatch": 1,
        "successful_transitions": 3,
        "sut_cases": 4,
        "sut_dispatch_total": 3,
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
    _read_pinned(
        v3_dir / "ha_acceptance.json",
        PINNED_V3_HA_ACCEPTANCE_SHA256,
        "archived v3 Home Assistant acceptance",
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
        == PINNED_V3_HA_ACCEPTANCE_SHA256,
        "v3/HA binding changed",
    )

    bindings = _mapping(
        validation.get("source_bindings"),
        "v3 source bindings",
    )
    expected_sources = {
        "clarify_commit.py": v3_dir / "code" / "clarify_commit.py",
        "tests/test_clarify_commit.py": v3_dir / "code" / "tests" / "test_clarify_commit.py",
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


def _verify_v4(case_dir: Path) -> dict[str, object]:
    """Verify the final closure while preserving all earlier classifications."""

    validation_payload = _read_pinned(
        case_dir / "evidence" / "v4" / "validation.json",
        PINNED_V4_VALIDATION_SHA256,
        "v4 validation",
    )
    validation = _load_json(validation_payload, "v4 validation")
    _require(
        set(validation)
        == {
            "analysis_classification",
            "clean_room",
            "evidence_version",
            "home_assistant_acceptance",
            "implementation_commit",
            "immutable_evidence",
            "independent_review",
            "non_claims",
            "repository_hygiene",
            "reproduction_results",
            "schema_version",
            "source_bindings",
            "status",
            "validation_date",
            "validation_results",
        },
        "v4 top-level field set changed",
    )
    _require(validation.get("schema_version") == 1, "v4 validation schema changed")
    _require(validation.get("status") == "validated", "v4 validation did not pass")
    _require(
        validation.get("evidence_version") == "v4-submission-readiness-closure",
        "v4 evidence version changed",
    )
    _require(
        validation.get("validation_date") == "2026-08-27",
        "v4 validation date changed",
    )

    classification = _mapping(
        validation.get("analysis_classification"),
        "v4 analysis classification",
    )
    _require(
        classification
        == {
            "confirmatory": False,
            "held_out": False,
            "home_assistant_acceptance_rerun": True,
            "model_rerun": False,
            "official_v2_replay": False,
            "stage": "final post-review policy and real-Home-Assistant acceptance closure",
            "v1_remains_sole_formal": True,
            "v2_record_remains_immutable": True,
            "v3_record_remains_immutable": True,
        },
        "v4 analysis classification changed",
    )
    commit = _mapping(
        validation.get("implementation_commit"),
        "v4 implementation commit",
    )
    _require(
        commit
        == {
            "sha": PINNED_V4_IMPLEMENTATION_COMMIT,
            "signed_off_by": (
                "MittaPei <315415437+MittaPei@users.noreply.github.com>"
            ),
            "subject": "fix: close correction and state-drift gaps",
        },
        "v4 implementation commit changed",
    )
    immutable = _mapping(
        validation.get("immutable_evidence"),
        "v4 immutable evidence",
    )
    _require(
        immutable
        == {
            "home_assistant_acceptance_sha256": PINNED_V4_HA_ACCEPTANCE_SHA256,
            "v1_manifest_sha256": PINNED_V1_MANIFEST_SHA256,
            "v2_manifest_sha256": PINNED_V2_MANIFEST_SHA256,
            "v3_validation_sha256": PINNED_V3_VALIDATION_SHA256,
        },
        "v4 immutable-evidence binding changed",
    )

    source_groups = _mapping(validation.get("source_bindings"), "v4 source groups")
    _require(
        set(source_groups) == {"implementation", "presentation", "validation_harness"},
        "v4 source-binding groups changed",
    )
    expected_groups = {
        "implementation": V4_IMPLEMENTATION_SOURCE_FILES,
        "presentation": V4_PRESENTATION_FILES,
        "validation_harness": V4_VALIDATION_HARNESS_FILES,
    }
    for group_name, source_files in expected_groups.items():
        bindings = _mapping(source_groups.get(group_name), f"v4 {group_name} bindings")
        _require(
            set(bindings) == set(source_files),
            f"v4 {group_name} source-binding set changed",
        )
        for relative in source_files:
            source_path = case_dir / V4_ARCHIVED_SOURCE_PATHS.get(relative, relative)
            _verify_binding(
                _read_bytes(source_path, f"v4 source {relative}"),
                bindings[relative],
                f"v4 source {relative}",
            )

    expected_results = {
        "artifact_verifier": {
            "command": "python verify_artifacts.py",
            "result": "passed",
        },
        "case_full_suite": {
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q"
            ),
            "passed": 188,
            "result": "passed",
        },
        "diff_check": {
            "command": "git diff --check && git diff --cached --check",
            "result": "passed",
        },
        "home_assistant_suite": {
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 python -m unittest "
                "tests/test_ha_acceptance.py -q"
            ),
            "passed": 13,
            "result": "passed",
        },
        "policy_suite": {
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 python -m unittest "
                "tests/test_clarify_commit.py -q"
            ),
            "passed": 108,
            "result": "passed",
        },
        "python_compile": {
            "command": "python -m py_compile *.py tests/*.py",
            "result": "passed",
        },
        "real_home_assistant": {
            "case_count": 4,
            "command": (
                "timeout 900 python ha_acceptance.py --output "
                "evidence/ha_acceptance.json"
            ),
            "drift_sut_dispatch_delta": 0,
            "rejected_before_dispatch": 1,
            "result": "passed",
            "successful_transitions": 3,
            "sut_dispatch_total": 3,
            "task_resources_after": 0,
        },
        "ruff": {
            "command": (
                "python -m ruff check clarify_commit.py ha_acceptance.py "
                "reproduce_v1.py reproduce_v2.py verify_artifacts.py tests"
            ),
            "result": "passed",
        },
    }
    results = _mapping(validation.get("validation_results"), "v4 validation results")
    _require(set(results) == set(expected_results), "v4 validation-result set changed")
    for name, expected in expected_results.items():
        _require(
            _mapping(results.get(name), f"v4 validation result {name}") == expected,
            (
                "v4 full-suite result changed"
                if name == "case_full_suite"
                else f"v4 validation result changed: {name}"
            ),
        )

    clean_room = _mapping(validation.get("clean_room"), "v4 clean-room result")
    _require(
        clean_room
        == {
            "clone_mode": "local --no-hardlinks, detached at the implementation commit",
            "dependency_install": False,
            "fresh_venv": True,
            "full_tests_passed": 188,
            "model_rerun": False,
            "network_used": False,
            "pip_check": "passed",
            "python_version": "3.12.12",
            "source_commit": PINNED_V4_IMPLEMENTATION_COMMIT,
            "temporary_directory_removed": True,
            "v1_reproduction": "byte_identical",
            "v2_reproduction": "byte_identical",
            "verifier": "passed",
            "worktree_dirty_entries_after": 0,
        },
        "v4 clean-room result changed",
    )
    hygiene = _mapping(
        validation.get("repository_hygiene"),
        "v4 repository hygiene",
    )
    _require(
        hygiene
        == {
            "lfs_pointer_files": 0,
            "max_tracked_file_threshold_bytes": 1048576,
            "model_weight_files": 0,
            "personal_absolute_path_matches": 0,
            "private_key_or_token_pattern_files": 0,
            "result": "passed",
            "runtime_sensitive_ha_json_keys": 0,
            "scope": (
                "all 52 tracked case files at the implementation commit and its "
                "technical change set"
            ),
            "tracked_files_over_threshold": 0,
            "unexpected_git_modes": 0,
        },
        "v4 repository-hygiene result changed",
    )

    expected_reproductions = {
        "v1": {
            "command": "python reproduce_v1.py",
            "expected_evaluator_exit_code": 1,
            "files": {
                "report.json": (
                    "edea57b50e0c9ea789ca97252ada87e7064d87b8ac739c0f019519441ed2be97"
                ),
                "trials.jsonl": (
                    "c7f0c97943bd49f4c21306eadeb38a69de8b063e0d821d21ee966c03ee287171"
                ),
            },
            "model_rerun": False,
            "quality_gate": "fail",
            "result": "byte_identical",
        },
        "v2": {
            "command": "python reproduce_v2.py",
            "expected_replay_exit_code": 1,
            "exploratory_gate": "fail",
            "files": {
                "manifest.json": PINNED_V2_MANIFEST_SHA256,
                "report.json": (
                    "b9df72353b2d20125ec75279e686e6316569dcc16b7b3db0666845a04452ebe9"
                ),
                "trials.jsonl": (
                    "e7a02a537ccf88afd101172ca2bafd8d3acc077b2030dc03b8ca7b5b1bf2ef5e"
                ),
            },
            "model_rerun": False,
            "raw_outputs_reused_from_v1": True,
            "result": "byte_identical",
        },
    }
    _require(
        _mapping(validation.get("reproduction_results"), "v4 reproductions")
        == expected_reproductions,
        "v4 reproduction result changed",
    )

    expected_ha = {
        "after_drift_state_digest": (
            "80e68977cce90ffd209341d5cdd5fb029c287f2cc719802fd12abbc6ec6a7f06"
        ),
        "case_count": 4,
        "drift_sut_dispatch_delta": 0,
        "evidence_generated_after_bound_sources": True,
        "external_fault_injection": 1,
        "image": "ghcr.io/home-assistant/home-assistant:2026.8.3",
        "prepared_state_digest": (
            "bf343f777d3bbb8d7dd36a3d821f5e3c2f206a0d4f7fcfe54da9b043ec6ce7a4"
        ),
        "rejected_before_dispatch": 1,
        "result": "passed",
        "schema_version": 2,
        "setup_direct_rest": 4,
        "successful_transitions": 3,
        "sut_dispatch_total": 3,
        "total_service_calls": 8,
    }
    ha_record = _mapping(
        validation.get("home_assistant_acceptance"),
        "v4 Home Assistant summary",
    )
    _require(ha_record == expected_ha, "v4 Home Assistant summary changed")
    archived_ha = _verify_v4_ha_archive(case_dir)
    _require(
        archived_ha.get("artifact_sha256") == PINNED_V4_HA_ACCEPTANCE_SHA256
        and archived_ha.get("sut_cases") == ha_record["case_count"]
        and archived_ha.get("successful_transitions")
        == ha_record["successful_transitions"]
        and archived_ha.get("rejected_before_dispatch")
        == ha_record["rejected_before_dispatch"]
        and archived_ha.get("sut_dispatch_total") == ha_record["sut_dispatch_total"]
        and archived_ha.get("drift_sut_dispatch_delta")
        == ha_record["drift_sut_dispatch_delta"],
        "v4/Home Assistant cross-check changed",
    )

    review = _mapping(validation.get("independent_review"), "v4 independent review")
    _require(
        review
        == {
            "blockers": 0,
            "major_findings": 0,
            "method": (
                "three isolated AI-assisted read-only reviews followed by "
                "main-agent verification"
            ),
            "review_passes": 3,
            "scopes": [
                "correction semantics and adversarial selector bridges",
                "real Home Assistant target-drift execution, accounting, and cleanup",
                (
                    "staged scope, immutable archives, privacy, credentials, and "
                    "repository hygiene"
                ),
            ],
        },
        "v4 independent-review result changed",
    )
    expected_non_claims = [
        "This validation is not a new model evaluation.",
        "It does not change, replace, or repair the recorded v1 or v2 metrics.",
        (
            "The v2 analysis reuses the v1 raw outputs and remains exploratory, "
            "not held-out or confirmatory."
        ),
        (
            "The Home Assistant acceptance uses fixed recorded Domux-output "
            "fixtures, not an uninterrupted live model-to-Home-Assistant run."
        ),
        (
            "Four acceptance cases mean three committed transitions plus one "
            "rejection before dispatch, not four committed transitions."
        ),
        (
            "The state-drift mutation is an out-of-band fault injection and is "
            "excluded from the SUT dispatch count."
        ),
        "The preview is an explanatory infographic, not a terminal screenshot.",
    ]
    _require(
        list(_sequence(validation.get("non_claims"), "v4 non-claims"))
        == expected_non_claims,
        "v4 non-claim set changed",
    )
    return {
        "status": "verified",
        "validation_sha256": PINNED_V4_VALIDATION_SHA256,
        "implementation_commit": PINNED_V4_IMPLEMENTATION_COMMIT,
        "policy_tests": 108,
        "ha_tests": 13,
        "full_tests": 188,
        "clean_room_tests": 188,
        "model_rerun": False,
        "official_v2_replay": False,
    }


def verify_all(case_dir: Path = CASE_DIR) -> dict[str, object]:
    """Verify all immutable evidence and return a deterministic summary."""

    case_dir = Path(case_dir)
    _, evaluation = _verify_frozen_data(case_dir)
    v1 = _verify_v1(case_dir, evaluation)
    diagnostic = _verify_v1_first_match_diagnostic(case_dir, evaluation)
    v2 = _verify_v2(case_dir, evaluation)
    ha = _verify_ha(case_dir)
    v3 = _verify_v3(case_dir)
    v4 = _verify_v4(case_dir)
    return {
        "status": "verified",
        "v1": v1,
        "post_formal_diagnostic": diagnostic,
        "v2": v2,
        "v3": v3,
        "v4": v4,
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
