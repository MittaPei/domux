#!/usr/bin/env python3
"""Run a post-formal naive first-match diagnostic over frozen v1 evidence.

This script is deliberately separate from ``evaluate.py``.  It does not add an
arm to the pre-registered B0/B1/B2 protocol, make model calls, or rewrite any
formal artifact.  It asks a narrower counterfactual question: what would have
happened if every structurally valid Domux instruction had selected the first
matching entity in the already-frozen inventory order and dispatched directly?

The frozen v1 parser, planner, and in-memory adapter are loaded from the source
archive bound by ``evidence/v1/manifest.json``.  Every input is checked against
its pre-existing SHA-256 before the diagnostic runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = CASE_DIR / "data" / "scenarios.jsonl"
DEFAULT_RAW_EVIDENCE = CASE_DIR / "evidence" / "v1" / "domux_raw.jsonl"
DEFAULT_FORMAL_REPORT = CASE_DIR / "evidence" / "v1" / "report.json"
DEFAULT_V1_MANIFEST = CASE_DIR / "evidence" / "v1" / "manifest.json"
DEFAULT_V1_POLICY = CASE_DIR / "evidence" / "v1" / "code" / "clarify_commit.py"
DEFAULT_OUTPUT = CASE_DIR / "evidence" / "diagnostics" / "v1_first_match.json"

EXPECTED_SHA256 = {
    "dataset": "0e27842c62d9cd4e4b1467b43e3ebcd346c79c0125c4f40cce97d363c821a0a0",
    "raw_evidence": "c0561bc72042dc7415d322fea90649866355dc44d2547f246d87cd87d367e966",
    "formal_report": "edea57b50e0c9ea789ca97252ada87e7064d87b8ac739c0f019519441ed2be97",
    "v1_manifest": "5f1c842676a367a9b5ae2cd948239a4f111bf0498e3cc916b57239ea671a9396",
    "v1_policy": "33d6616b341469f545dc9a7c01d1acaaff35fe7d42d9cefb5741170aad966e10",
}
EXPECTED_BASE_COUNT = 48
EXPECTED_PROBE_COUNT = 96
VARIANTS = ("clear", "ambiguous")
SLOTS = ("action", "device", "attribute", "value", "unit", "room", "floor")


class DiagnosticInputError(ValueError):
    """A bound v1 input no longer has its frozen content or schema."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DiagnosticInputError(f"cannot read bound input: {path.name}") from exc


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise DiagnosticInputError(f"{label} SHA-256 differs from frozen v1")
    return observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticInputError(f"cannot read valid {label} JSON") from exc
    if not isinstance(value, dict):
        raise DiagnosticInputError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticInputError(f"cannot read {label} JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DiagnosticInputError(f"{label} line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosticInputError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise DiagnosticInputError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _load_v1_policy(path: Path) -> ModuleType:
    module_name = f"_domux_v1_first_match_{_sha256_text(str(path.resolve()))[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticInputError("cannot load the frozen v1 policy module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validated_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        base_id = item.get("base_id")
        split = item.get("split")
        if not isinstance(base_id, str) or not base_id:
            raise DiagnosticInputError("scenario base_id must be a non-empty string")
        if base_id in seen:
            raise DiagnosticInputError(f"duplicate scenario base_id: {base_id}")
        seen.add(base_id)
        if split == "eval":
            for field in (
                "ambiguous_command",
                "expected_target_entity",
                "expected_delta",
                "initial_states",
                "inventory",
                "category",
            ):
                if field not in item:
                    raise DiagnosticInputError(
                        f"evaluation scenario {base_id} is missing {field}"
                    )
            eval_rows.append(dict(item))
        elif split != "dev":
            raise DiagnosticInputError(f"scenario {base_id} has an invalid split")
    if len(eval_rows) != EXPECTED_BASE_COUNT:
        raise DiagnosticInputError("dataset must contain exactly 48 evaluation bases")
    return eval_rows


def _validated_evidence(
    items: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], str] = {}
    for row in eval_rows:
        base_id = str(row["base_id"])
        for variant in VARIANTS:
            command = row.get(f"{variant}_command")
            if not isinstance(command, str) or not command:
                raise DiagnosticInputError(
                    f"scenario {base_id} has an invalid {variant} command"
                )
            expected[(base_id, variant)] = command

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        base_id = item.get("base_id")
        variant = item.get("variant")
        key = (str(base_id), str(variant))
        if key not in expected:
            raise DiagnosticInputError(f"unexpected frozen evidence key: {key}")
        if key in indexed:
            raise DiagnosticInputError(f"duplicate frozen evidence key: {key}")
        command = item.get("command")
        raw_output = item.get("raw_output")
        if command != expected[key]:
            raise DiagnosticInputError(f"frozen command mismatch: {key}")
        if item.get("query_sha256") != _sha256_text(str(command)):
            raise DiagnosticInputError(f"frozen query hash mismatch: {key}")
        if item.get("status") != "ok":
            raise DiagnosticInputError(f"v1 probe is not successful: {key}")
        if not isinstance(raw_output, str) or not raw_output:
            raise DiagnosticInputError(f"v1 raw output is missing: {key}")
        if item.get("raw_output_sha256") != _sha256_text(raw_output):
            raise DiagnosticInputError(f"frozen raw-output hash mismatch: {key}")
        indexed[key] = dict(item)
    if len(items) != EXPECTED_PROBE_COUNT or set(indexed) != set(expected):
        raise DiagnosticInputError("v1 evidence must contain all 96 paired probes")
    return indexed


def _entity_inventory(policy: ModuleType, row: Mapping[str, object]) -> tuple[Any, ...]:
    inventory = row.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise DiagnosticInputError("scenario inventory must be a non-empty list")
    entities: list[Any] = []
    for item in inventory:
        if not isinstance(item, dict):
            raise DiagnosticInputError("inventory entries must be objects")
        try:
            entities.append(policy.EntitySpec(
                entity_id=str(item["entity_id"]),
                domain=str(item["domain"]),
                device=str(item["device"]),
                room=str(item["room"]),
                floor=str(item["floor"]),
                aliases=tuple(str(alias) for alias in item.get("aliases", ())),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise DiagnosticInputError("scenario inventory is invalid") from exc
    return tuple(entities)


def _snapshot(policy: ModuleType, adapter: Any, registry: Any) -> dict[str, object]:
    return {
        entity.entity_id: policy.controlled_projection(
            adapter.get_state(entity.entity_id), entity.domain
        )
        for entity in registry.entities
    }


def _expected_projection(
    policy: ModuleType,
    row: Mapping[str, object],
    registry: Any,
    which: str,
) -> dict[str, object]:
    target = str(row["expected_target_entity"])
    expected_delta = row.get("expected_delta")
    if not isinstance(expected_delta, dict) or which not in expected_delta:
        raise DiagnosticInputError("scenario expected_delta is invalid")
    return policy.controlled_projection(
        expected_delta[which], registry.get(target).domain
    )


def _wrong_target_transition(
    target: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    calls: Sequence[Mapping[str, object]],
) -> bool:
    if any(str(call.get("data", {}).get("entity_id")) != target for call in calls):
        return True
    return any(before[entity_id] != after[entity_id] for entity_id in before if entity_id != target)


def _ordered_candidates(
    registry: Any,
    inventory: Sequence[Any],
    instruction: Any,
) -> tuple[Any, ...]:
    """Match with v1 semantics but preserve the pre-frozen inventory order."""

    matching_ids = {
        entity.entity_id for entity in registry.candidates(instruction, context=None)
    }
    return tuple(entity for entity in inventory if entity.entity_id in matching_ids)


def _safe_error(exc: Exception) -> dict[str, str]:
    # Frozen planner errors contain only operation fields and fixed validation
    # text.  Keep the type and message for reproducibility without tracebacks or
    # local paths.
    return {"error_type": type(exc).__name__, "error": str(exc)}


def _run_trial(
    policy: ModuleType,
    row: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    inventory = _entity_inventory(policy, row)
    registry = policy.EntityRegistry(inventory)
    initial_states = row.get("initial_states")
    if not isinstance(initial_states, dict):
        raise DiagnosticInputError("scenario initial_states must be an object")
    adapter = policy.InMemoryHAAdapter(initial_states)
    before = _snapshot(policy, adapter, registry)
    raw_output = str(evidence["raw_output"])
    try:
        instructions = policy.parse_domux_output(raw_output)
    except policy.ParseError as exc:
        raise DiagnosticInputError(
            f"v1 ambiguous output is not structurally parseable: {row['base_id']}"
        ) from exc

    instruction_records: list[dict[str, object]] = []
    for index, instruction in enumerate(instructions, start=1):
        candidates = _ordered_candidates(registry, inventory, instruction)
        record: dict[str, object] = {
            "index": index,
            "instruction": instruction.to_pipe(),
            "candidate_ids_in_frozen_inventory_order": [
                candidate.entity_id for candidate in candidates
            ],
            "selected_entity_id": candidates[0].entity_id if candidates else None,
        }
        if not candidates:
            record["outcome"] = "not_dispatched_no_candidate"
            instruction_records.append(record)
            continue

        chosen = candidates[0]
        try:
            plan = policy.build_plan(
                instruction,
                chosen,
                adapter.get_state(chosen.entity_id),
            )
            receipt = adapter.call_service(plan.domain, plan.service, plan.service_data)
            postcondition = policy.projection_matches(
                policy.controlled_projection(receipt.after, plan.domain),
                plan.expected_projection,
            )
        except (policy.GroundingError, KeyError, TypeError, ValueError) as exc:
            record.update({
                "outcome": "not_dispatched_non_executable",
                **_safe_error(exc),
            })
            instruction_records.append(record)
            continue

        record.update({
            "outcome": "dispatched",
            "domain": plan.domain,
            "service": plan.service,
            "postcondition_match": bool(postcondition),
        })
        instruction_records.append(record)

    after = _snapshot(policy, adapter, registry)
    calls = tuple(adapter.sut_calls)
    target = str(row["expected_target_entity"])
    wrong_target = _wrong_target_transition(target, before, after, calls)
    exact_delta = bool(
        before[target] == _expected_projection(policy, row, registry, "before")
        and after[target] == _expected_projection(policy, row, registry, "after")
        and len(calls) == 1
        and str(calls[0].get("data", {}).get("entity_id")) == target
        and not wrong_target
        and all(
            record.get("postcondition_match") is True
            for record in instruction_records
            if record["outcome"] == "dispatched"
        )
    )
    if exact_delta:
        classification = "exact_expected_delta"
    elif wrong_target:
        classification = "wrong_target_transition"
    else:
        classification = "non_exact_without_wrong_target_transition"

    return {
        "base_id": str(row["base_id"]),
        "category": str(row["category"]),
        "variant": "ambiguous",
        "command": str(row["ambiguous_command"]),
        "query_sha256": str(evidence["query_sha256"]),
        "raw_output": raw_output,
        "raw_output_sha256": str(evidence["raw_output_sha256"]),
        "expected_target_entity": target,
        "instruction_count": len(instructions),
        "instructions": instruction_records,
        "sut_call_count": len(calls),
        "sut_calls": [
            {
                "domain": str(call["domain"]),
                "service": str(call["service"]),
                "entity_id": str(call["data"]["entity_id"]),
            }
            for call in calls
        ],
        "exact_delta_success": exact_delta,
        "wrong_target_transition": wrong_target,
        "classification": classification,
    }


def _count_metric(values: Iterable[bool]) -> dict[str, object]:
    materialized = tuple(bool(value) for value in values)
    denominator = len(materialized)
    if denominator == 0:
        raise ValueError("diagnostic metric cannot have an empty denominator")
    successes = sum(materialized)
    return {
        "successes": successes,
        "denominator": denominator,
        "rate": successes / denominator,
    }


def _formal_metric(
    formal_b0: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    value = formal_b0.get(name)
    if not isinstance(value, dict):
        raise DiagnosticInputError(f"formal B0 report is missing {name}")
    successes = value.get("successes")
    denominator = value.get("denominator")
    rate = value.get("rate")
    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or denominator != EXPECTED_BASE_COUNT
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isclose(
            float(rate),
            successes / EXPECTED_BASE_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise DiagnosticInputError(f"formal B0 metric is invalid: {name}")
    return {
        "successes": successes,
        "denominator": denominator,
        "rate": float(rate),
    }


def build_diagnostic(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    raw_evidence_path: Path = DEFAULT_RAW_EVIDENCE,
    formal_report_path: Path = DEFAULT_FORMAL_REPORT,
    manifest_path: Path = DEFAULT_V1_MANIFEST,
    policy_path: Path = DEFAULT_V1_POLICY,
    diagnostic_code_path: Path | None = None,
) -> dict[str, object]:
    paths = {
        "dataset": dataset_path,
        "raw_evidence": raw_evidence_path,
        "formal_report": formal_report_path,
        "v1_manifest": manifest_path,
        "v1_policy": policy_path,
    }
    observed_hashes = {
        label: _require_hash(path, EXPECTED_SHA256[label], label)
        for label, path in paths.items()
    }

    manifest = _read_json(manifest_path, "v1 manifest")
    if (
        manifest.get("evidence_version") != "v1-formal"
        or manifest.get("status") != "frozen"
        or manifest.get("artifacts", {}).get("domux_raw.jsonl")
        != observed_hashes["raw_evidence"]
        or manifest.get("artifacts", {}).get("report.json")
        != observed_hashes["formal_report"]
        or manifest.get("code", {}).get("clarify_commit.py")
        != observed_hashes["v1_policy"]
    ):
        raise DiagnosticInputError("v1 manifest bindings are invalid")

    rows = _validated_rows(_read_jsonl(dataset_path, "scenario dataset"))
    evidence = _validated_evidence(
        _read_jsonl(raw_evidence_path, "v1 raw evidence"), rows
    )
    formal_report = _read_json(formal_report_path, "formal v1 report")
    try:
        formal_b0 = formal_report["metrics"]["execution"]["B0_unique_or_abstain"]
    except (KeyError, TypeError) as exc:
        raise DiagnosticInputError("formal v1 report has no B0 execution metrics") from exc
    if not isinstance(formal_b0, dict):
        raise DiagnosticInputError("formal v1 B0 execution metrics must be an object")

    policy = _load_v1_policy(policy_path)
    trials = [
        _run_trial(policy, row, evidence[(str(row["base_id"]), "ambiguous")])
        for row in rows
    ]
    if len({str(trial["base_id"]) for trial in trials}) != EXPECTED_BASE_COUNT:
        raise DiagnosticInputError("diagnostic trials are not one-to-one with evaluation bases")

    formal_metrics = {
        "exact_delta_success": _formal_metric(
            formal_b0, "ambiguous_clean_exact_delta_success"
        ),
        "dispatch_coverage": _formal_metric(formal_b0, "dispatch_coverage"),
        "wrong_target_transition": _formal_metric(
            formal_b0, "wrong_target_transition_rate"
        ),
        "safe_abstention": _formal_metric(formal_b0, "safe_abstention_rate"),
    }
    diagnostic_metrics = {
        "structurally_parseable_output": _count_metric(
            trial["instruction_count"] >= 1 for trial in trials
        ),
        "bases_with_any_sut_call": _count_metric(
            trial["sut_call_count"] > 0 for trial in trials
        ),
        "formal_equivalent_dispatch_coverage": _count_metric(
            trial["sut_call_count"] == 1 for trial in trials
        ),
        "exact_delta_success": _count_metric(
            bool(trial["exact_delta_success"]) for trial in trials
        ),
        "wrong_target_transition": _count_metric(
            bool(trial["wrong_target_transition"]) for trial in trials
        ),
        "multiple_sut_calls": _count_metric(trial["sut_call_count"] > 1 for trial in trials),
        "order_sensitive_candidate_set": _count_metric(
            any(
                len(instruction["candidate_ids_in_frozen_inventory_order"]) > 1
                for instruction in trial["instructions"]
            )
            for trial in trials
        ),
    }
    diagnostic_metrics.update({
        "total_sut_calls": sum(int(trial["sut_call_count"]) for trial in trials),
        "instruction_outcomes": dict(sorted(Counter(
            str(instruction["outcome"])
            for trial in trials
            for instruction in trial["instructions"]
        ).items())),
        "classifications": dict(sorted(Counter(
            str(trial["classification"]) for trial in trials
        ).items())),
    })

    by_category: dict[str, object] = {}
    for category in sorted({str(trial["category"]) for trial in trials}):
        category_trials = [trial for trial in trials if trial["category"] == category]
        by_category[category] = {
            "bases": len(category_trials),
            "bases_with_any_sut_call": sum(
                trial["sut_call_count"] > 0 for trial in category_trials
            ),
            "exact_delta_successes": sum(
                bool(trial["exact_delta_success"]) for trial in category_trials
            ),
            "wrong_target_transitions": sum(
                bool(trial["wrong_target_transition"]) for trial in category_trials
            ),
        }

    code_path = Path(__file__) if diagnostic_code_path is None else diagnostic_code_path
    return {
        "schema_version": 1,
        "diagnostic_id": "v1-post-formal-naive-first-match",
        "status": "complete",
        "analysis_class": "post_formal_diagnostic_only",
        "formal_protocol_changed": False,
        "formal_metrics_changed": False,
        "model_calls": 0,
        "generation": {
            "method": "deterministic replay of hash-bound frozen inputs; no timestamps",
            "command": "python diagnose_first_match_v1.py",
        },
        "scope": {
            "source": "the 48 ambiguous probes in immutable v1 raw evidence",
            "independent_unit": "base_id",
            "base_count": EXPECTED_BASE_COUNT,
            "variant": "ambiguous",
            "mutations": "none; clean initial state only",
        },
        "counterfactual_policy": {
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
        "input_bindings": {
            "dataset": {
                "path": "data/scenarios.jsonl",
                "sha256": observed_hashes["dataset"],
            },
            "raw_evidence": {
                "path": "evidence/v1/domux_raw.jsonl",
                "sha256": observed_hashes["raw_evidence"],
            },
            "formal_report": {
                "path": "evidence/v1/report.json",
                "sha256": observed_hashes["formal_report"],
            },
            "v1_manifest": {
                "path": "evidence/v1/manifest.json",
                "sha256": observed_hashes["v1_manifest"],
            },
            "v1_policy": {
                "path": "evidence/v1/code/clarify_commit.py",
                "sha256": observed_hashes["v1_policy"],
            },
            "diagnostic_code": {
                "path": "diagnose_first_match_v1.py",
                "sha256": _sha256_file(code_path),
            },
        },
        "comparison": {
            "formal_arm": "B0_unique_or_abstain",
            "formal_v1": formal_metrics,
            "post_formal_first_match": diagnostic_metrics,
            "count_deltas_first_match_minus_formal": {
                "formal_equivalent_dispatches": (
                    diagnostic_metrics["formal_equivalent_dispatch_coverage"]["successes"]
                    - formal_metrics["dispatch_coverage"]["successes"]
                ),
                "exact_delta_successes": (
                    diagnostic_metrics["exact_delta_success"]["successes"]
                    - formal_metrics["exact_delta_success"]["successes"]
                ),
                "wrong_target_transitions": (
                    diagnostic_metrics["wrong_target_transition"]["successes"]
                    - formal_metrics["wrong_target_transition"]["successes"]
                ),
            },
            "by_category": by_category,
        },
        "limitations": [
            "This arm was specified after formal v1 inspection and is diagnostic, not pre-registered evidence.",
            "Inventory order is deterministic and pre-frozen but semantically arbitrary; first-match outcomes are order-dependent.",
            "The fixed balanced conformance suite does not estimate production prevalence or risk.",
            "The diagnostic uses the frozen in-memory adapter, not the separate real Home Assistant acceptance path.",
            "The result does not replace, rename, or enter any formal B0/B1/B2 metric or significance test.",
        ],
        "trials": trials,
    }


def render_diagnostic(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output is byte-identical instead of writing it",
    )
    args = parser.parse_args(argv)
    rendered = render_diagnostic(build_diagnostic())
    if args.check:
        try:
            observed = args.output.read_bytes()
        except OSError as exc:
            raise DiagnosticInputError("cannot read committed diagnostic output") from exc
        if observed != rendered:
            raise DiagnosticInputError("committed diagnostic output is stale")
        print(f"first-match diagnostic verified: {_sha256_bytes(rendered)}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    print(f"first-match diagnostic written: {_sha256_bytes(rendered)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticInputError as exc:
        print(f"first-match diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
