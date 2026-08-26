#!/usr/bin/env python3
"""Deterministic grounding and one-time commit for Domux seven-slot outputs.

The model remains a parser.  This module deliberately keeps inventory lookup,
clarification, authorization lifetime, state binding, dispatch and postcondition
checks outside the model so every boundary can be audited and replayed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence


SLOTS = ("action", "device", "attribute", "value", "unit", "room", "floor")
SUPPORTED_DOMAINS = frozenset({"light", "cover", "climate"})
ENTITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$")
GENERIC_DEVICE_ALIASES = {
    "light": "light",
    "lamp": "light",
    "lighting": "light",
    "curtain": "cover",
    "curtains": "cover",
    "blind": "cover",
    "blinds": "cover",
    "shade": "cover",
    "ac": "climate",
    "a c": "climate",
    "air conditioner": "climate",
    "air conditioning": "climate",
}
COLOR_RGB = {
    "blue": [0, 0, 255],
    "cyan": [0, 255, 255],
    "cool white": [201, 226, 255],
    "green": [0, 128, 0],
    "lavender": [230, 230, 250],
    "magenta": [255, 0, 255],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "purple": [128, 0, 128],
    "red": [255, 0, 0],
    "sky blue": [135, 206, 235],
    "warm white": [255, 244, 229],
    "white": [255, 255, 255],
    "yellow": [255, 255, 0],
}


class ParseError(ValueError):
    """The raw Domux text is not a bounded sequence of seven-slot records."""


class GroundingError(ValueError):
    """A structurally valid model output cannot be mapped to an executable plan."""


class AdapterError(RuntimeError):
    """The Home Assistant adapter could not read or change controlled state."""


class ServiceCallError(AdapterError):
    """A dispatch failure with action-local, non-heuristic outcome metadata."""

    def __init__(
        self,
        message: str,
        *,
        attempted: bool,
        acknowledged: bool,
        outcome_unknown: bool,
    ) -> None:
        super().__init__(message)
        self.attempted = attempted
        self.acknowledged = acknowledged
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class ServiceCallResult:
    after: Mapping[str, object]
    attempted: bool
    acknowledged: bool
    outcome_unknown: bool = False


def normalize_text(value: object) -> str:
    """Normalize only for comparison; never use this to overwrite raw evidence."""

    return " ".join(
        str(value)
        .replace("_", " ")
        .replace("-", " ")
        .replace("’", "'")
        .replace("‘", "'")
        .split()
    ).casefold()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DomuxInstruction:
    action: str
    device: str
    attribute: str
    value: str
    unit: str
    room: str
    floor: str

    @classmethod
    def from_fields(cls, fields: Sequence[str]) -> "DomuxInstruction":
        if len(fields) != len(SLOTS):
            raise ParseError(f"expected seven fields, got {len(fields)}")
        cleaned = tuple(field.strip() for field in fields)
        if any(not field for field in cleaned):
            raise ParseError("empty fields must be represented by '*'")
        if not cleaned[0] or cleaned[0] == "*":
            raise ParseError("action cannot be omitted")
        return cls(*cleaned)

    def to_pipe(self) -> str:
        return "|".join(getattr(self, slot) for slot in SLOTS)

    def canonical_slots(self) -> dict[str, str]:
        return {slot: normalize_text(getattr(self, slot)) for slot in SLOTS}


def parse_domux_output(raw_output: str, *, max_instructions: int = 8) -> tuple[DomuxInstruction, ...]:
    """Parse raw model text without silently dropping malformed segments."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ParseError("model output is empty")
    if any(ord(char) < 32 and char not in "\r\n\t" for char in raw_output):
        raise ParseError("model output contains control characters")
    segments = [segment.strip() for segment in raw_output.replace("&", "\n").splitlines()]
    if any(not segment for segment in segments):
        segments = [segment for segment in segments if segment]
    if not segments:
        raise ParseError("model output has no non-empty instruction")
    if len(segments) > max_instructions:
        raise ParseError(f"model output has more than {max_instructions} instructions")
    parsed: list[DomuxInstruction] = []
    for index, segment in enumerate(segments, start=1):
        fields = segment.split("|")
        if len(fields) != len(SLOTS):
            raise ParseError(f"instruction {index} has {len(fields)} fields, expected seven")
        parsed.append(DomuxInstruction.from_fields(fields))
    return tuple(parsed)


@dataclass(frozen=True)
class EntitySpec:
    entity_id: str
    domain: str
    device: str
    room: str
    floor: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"unsupported entity domain: {self.domain}")
        if not self.entity_id.startswith(f"{self.domain}."):
            raise ValueError(f"entity_id/domain mismatch: {self.entity_id}")
        if not ENTITY_ID_RE.fullmatch(self.entity_id):
            raise ValueError(f"invalid Home Assistant entity_id: {self.entity_id}")

    def stable_metadata(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "device": self.device,
            "room": self.room,
            "floor": self.floor,
            "aliases": sorted(self.aliases, key=lambda value: (normalize_text(value), value)),
        }


@dataclass(frozen=True)
class SessionContext:
    recent_entity_ids: tuple[str, ...] = ()


class EntityRegistry:
    """A deterministic, immutable view of the allowed Home Assistant entities."""

    def __init__(self, entities: Iterable[EntitySpec]):
        by_id: dict[str, EntitySpec] = {}
        for entity in entities:
            if entity.entity_id in by_id:
                raise ValueError(f"duplicate entity_id: {entity.entity_id}")
            by_id[entity.entity_id] = entity
        if not by_id:
            raise ValueError("registry cannot be empty")
        self._by_id = by_id

    def get(self, entity_id: str) -> EntitySpec:
        try:
            return self._by_id[entity_id]
        except KeyError as exc:
            raise GroundingError(f"entity is not in the allowed registry: {entity_id}") from exc

    @property
    def entities(self) -> tuple[EntitySpec, ...]:
        return tuple(sorted(self._by_id.values(), key=self._sort_key))

    @staticmethod
    def _sort_key(entity: EntitySpec) -> tuple[str, str, str, str]:
        return tuple(normalize_text(part) for part in (
            entity.floor, entity.room, entity.device, entity.entity_id,
        ))

    @staticmethod
    def _domain_hint(instruction: DomuxInstruction) -> str | None:
        device = normalize_text(instruction.device)
        if device in GENERIC_DEVICE_ALIASES:
            return GENERIC_DEVICE_ALIASES[device]
        attribute = normalize_text(instruction.attribute)
        if attribute in {"brightness", "color", "color temperature", "colortemperature"}:
            return "light"
        if attribute in {"position", "openness"}:
            return "cover"
        if attribute in {"temperature", "mode", "wind speed", "windspeed", "fan speed"}:
            return "climate"
        return None

    @staticmethod
    def _device_matches(entity: EntitySpec, requested: str) -> bool:
        requested_norm = normalize_text(requested)
        if requested_norm in {"*", "it", "that", "that one", "the other", "other"}:
            return True
        names = {normalize_text(entity.device), *(normalize_text(alias) for alias in entity.aliases)}
        if requested_norm in names:
            return True
        domain = GENERIC_DEVICE_ALIASES.get(requested_norm)
        return domain == entity.domain

    def candidates(
        self,
        instruction: DomuxInstruction,
        context: SessionContext | None = None,
    ) -> tuple[EntitySpec, ...]:
        domain_hint = self._domain_hint(instruction)
        room = normalize_text(instruction.room)
        floor = normalize_text(instruction.floor)
        candidates = [
            entity
            for entity in self._by_id.values()
            if (domain_hint is None or entity.domain == domain_hint)
            and self._device_matches(entity, instruction.device)
            and (room == "*" or normalize_text(entity.room) == room)
            and (floor == "*" or normalize_text(entity.floor) == floor)
        ]

        requested = normalize_text(instruction.device)
        if context and requested in {"*", "it", "that", "that one", "the other", "other"}:
            recent = set(context.recent_entity_ids)
            contextual = [entity for entity in candidates if entity.entity_id in recent]
            if contextual:
                candidates = contextual
        return tuple(sorted(candidates, key=self._sort_key))

    def metadata_digest(self, entity_ids: Iterable[str]) -> str:
        ordered = [self.get(entity_id).stable_metadata() for entity_id in sorted(set(entity_ids))]
        return digest_json(ordered)

    def with_replacement(self, entity: EntitySpec) -> "EntityRegistry":
        if entity.entity_id not in self._by_id:
            raise GroundingError(f"cannot replace unknown entity: {entity.entity_id}")
        updated = dict(self._by_id)
        updated[entity.entity_id] = entity
        return EntityRegistry(updated.values())


@dataclass(frozen=True)
class Clarification:
    required: bool
    reason: str
    candidates: tuple[EntitySpec, ...]
    prompt: str | None
    reasons: tuple[str, ...] = ()
    unresolved_slots: tuple[str, ...] = ()


def _candidate_option(index: int, candidate: EntitySpec) -> str:
    """Render a visibly unique, stable authorization choice."""

    aliases = sorted(
        {alias for alias in candidate.aliases if normalize_text(alias)},
        key=lambda value: (normalize_text(value), value),
    )
    alias_text = f" / alias: {', '.join(aliases)}" if aliases else ""
    return (
        f"{index}. {candidate.floor} / {candidate.room} / {candidate.device}"
        f"{alias_text} / id: {candidate.entity_id}"
    )


def clarification_for(candidates: Sequence[EntitySpec], *, max_display: int = 3) -> Clarification:
    if not candidates:
        return Clarification(True, "no_registry_match", (), None, ("no_registry_match",))
    if len(candidates) == 1:
        return Clarification(False, "unique_registry_match", tuple(candidates), None)
    displayed = tuple(candidates[:max_display])
    if len(candidates) > max_display:
        prompt = "Which room or floor should I use? More than three devices match."
        return Clarification(True, "too_many_candidates", displayed, prompt)
    options = "; ".join(
        _candidate_option(index, candidate)
        for index, candidate in enumerate(displayed, start=1)
    )
    return Clarification(
        True,
        "multiple_registry_matches",
        displayed,
        f"Which device: {options}?",
        ("multiple_registry_matches",),
    )


def _phrase_in(normalized_text: str, phrase: object) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase or normalized_phrase == "*":
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])",
        normalized_text,
    ) is not None


def _numbers_in(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?", text))


def _canonical_number_token(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".15g")


def _operation_value_token(value: str) -> str | None:
    if value == "*":
        return None
    try:
        return f"number:{_canonical_number_token(float(value))}"
    except ValueError:
        return f"value:{normalize_text(value)}"


def _value_is_explicitly_excluded(text: str, value: str) -> bool:
    """Bind a negative choice to the exact operation value it excludes."""

    token = normalize_text(value)
    if not token or token == "*":
        return False
    prefix = (
        r"(?:\b(?:anything|something|everything|any(?:\s+[a-z]+)?|a\s+value)\s+"
        r"(?:but|other\s+than|besides|apart\s+from)\s+(?:the\s+)?|"
        r"\b(?:other\s+than|except|besides|apart\s+from|avoid|without)\s+"
        r"(?:using\s+)?(?:the\s+)?|"
        r"\b(?:do\s+not|don't|dont|never)\s+(?:use|choose|select)\s+(?:the\s+)?|"
        r"\bnot\s+(?:the\s+)?)"
    )
    return re.search(
        rf"{prefix}{re.escape(token)}(?![a-z0-9])",
        normalize_text(text),
    ) is not None


def _has_operation_value_exclusion(text: str) -> bool:
    """Detect requests that permit a set of values but do not choose one."""

    normalized = normalize_text(text)
    if re.search(
        r"\b(?:anything|something|everything|any(?:\s+[a-z]+)?|a\s+value)\s+"
        r"(?:but|other\s+than|besides|apart\s+from)\b|"
        r"\b(?:do\s+not|don't|dont|never)\s+(?:use|choose|select)\b|"
        r"\b(?:set|change|make|use)\b.{0,40}\b"
        r"(?:avoid|without|other\s+than|besides|apart\s+from)\b",
        normalized,
    ):
        return True
    known_values = (
        *COLOR_RGB,
        "fan only", "heat cool", "cool", "heat", "dry", "fan", "auto",
        "low", "medium", "high",
    )
    return any(_value_is_explicitly_excluded(normalized, value) for value in known_values)


def _excluded_operation_value_tokens(
    text: str,
    source_instructions: Sequence[DomuxInstruction] = (),
) -> tuple[str, ...]:
    """Return audit metadata for values explicitly ruled out by the user."""

    known_values = (
        *COLOR_RGB,
        "fan only", "heat cool", "cool", "heat", "dry", "fan", "auto",
        "low", "medium", "high",
    )
    tokens = {
        token
        for value in known_values
        if _value_is_explicitly_excluded(text, value)
        if (token := _operation_value_token(value)) is not None
    }
    for number in _numbers_in(normalize_text(text)):
        rendered = _canonical_number_token(number)
        if _value_is_explicitly_excluded(text, rendered):
            tokens.add(f"number:{rendered}")
    for instruction in source_instructions:
        if _value_is_explicitly_excluded(text, instruction.value):
            token = _operation_value_token(instruction.value)
            if token is not None:
                tokens.add(token)
    return tuple(sorted(tokens))


def _value_supported(text: str, value: str) -> bool:
    if value == "*":
        return True
    normalized = normalize_text(text)
    try:
        wanted = float(value)
    except ValueError:
        return _phrase_in(normalized, value)
    if any(math.isclose(wanted, number, rel_tol=0, abs_tol=0.001) for number in _numbers_in(normalized)):
        return True
    return math.isclose(wanted, 50.0, rel_tol=0, abs_tol=0.001) and _phrase_in(normalized, "halfway")


def _attribute_supported(text: str, instruction: DomuxInstruction) -> bool:
    attribute = normalize_text(instruction.attribute)
    if attribute == "*":
        return True
    normalized = normalize_text(text)
    if attribute == "brightness":
        return any(_phrase_in(normalized, term) for term in ("brightness", "bright", "brighter", "dimmer")) or (
            _phrase_in(normalized, "percent")
            and any(_phrase_in(normalized, term) for term in ("light", "lamp"))
        )
    if attribute in {"position", "openness"}:
        return _phrase_in(normalized, "position") or _phrase_in(normalized, "openness") or (
            any(_phrase_in(normalized, term) for term in ("percent", "halfway", "open", "close", "move", "adjust"))
            and any(_phrase_in(normalized, term) for term in ("curtain", "blind", "shade"))
        ) or any(_phrase_in(normalized, term) for term in ("open", "close"))
    if attribute == "temperature":
        return any(_phrase_in(normalized, term) for term in (
            "temperature", "degree", "degrees", "celsius", "warmer", "cooler"
        )) or (
            bool(_numbers_in(normalized))
            and any(_phrase_in(normalized, term) for term in ("ac", "air conditioner", "air conditioning"))
        )
    if attribute == "colortemperature":
        return "color temperature" in normalized or _phrase_in(normalized, "kelvin") or bool(
            re.search(r"\d+(?:\.\d+)?\s*k\b", normalized)
        )
    if attribute == "color":
        return _phrase_in(normalized, "color") or any(
            _phrase_in(normalized, color) for color in COLOR_RGB
        )
    if attribute == "mode":
        return _phrase_in(normalized, "mode") or (
            any(_phrase_in(normalized, mode) for mode in ("cool", "heat", "dry", "fan", "auto"))
            and any(_phrase_in(normalized, term) for term in ("ac", "air conditioner", "air conditioning"))
        )
    if attribute in {"windspeed", "wind speed", "fan speed"}:
        return any(_phrase_in(normalized, term) for term in ("wind", "fan", "low", "medium", "high"))
    return False


def _attribute_supported_for_entity(
    text: str,
    instruction: DomuxInstruction,
    entity: EntitySpec,
) -> bool:
    """Allow registry-domain inference only when the operation value is explicit."""

    if _attribute_supported(text, instruction):
        return True
    attribute = normalize_text(instruction.attribute)
    if attribute == "*":
        return True
    value_supported = _value_supported(text, instruction.value)
    unit_supported = _unit_supported(text, instruction)
    normalized = normalize_text(text)
    if entity.domain == "cover" and attribute in {"position", "openness"}:
        return value_supported and unit_supported and normalize_text(instruction.unit) == "percent"
    if entity.domain == "light" and attribute == "brightness":
        return value_supported and unit_supported and normalize_text(instruction.unit) == "percent"
    if entity.domain == "light" and attribute == "colortemperature":
        return value_supported and unit_supported and normalize_text(instruction.unit) == "kelvin"
    if entity.domain == "light" and attribute == "color":
        return value_supported and any(_phrase_in(normalized, color) for color in COLOR_RGB)
    if entity.domain == "climate" and attribute == "temperature":
        return value_supported and unit_supported and normalize_text(instruction.unit) == "celsius"
    if entity.domain == "climate" and attribute == "mode":
        return value_supported and any(
            _phrase_in(normalized, mode)
            for mode in ("cool", "heat", "dry", "fan only", "fan", "auto")
        )
    if entity.domain == "climate" and attribute in {"windspeed", "wind speed", "fan speed"}:
        return value_supported and unit_supported and normalize_text(instruction.unit) == "level"
    return False


def _action_supported(text: str, instruction: DomuxInstruction) -> bool:
    normalized = normalize_text(text)
    if re.search(
        r"\b(?:do\s+not|don't|never)\s+(?:turn|switch|open|close|set|change|make|adjust|raise|lower)\b",
        normalized,
    ):
        return False
    action = normalize_text(instruction.action)
    directional = _directional_actions(normalized)
    # A proposed ``set`` must not bypass opposing open/close or on/off clauses.
    # Such clauses describe more than one state transition and need an explicit
    # clarification even when the model happens to emit a value-bearing action.
    if len(directional) > 1:
        return False
    if action == "turnon":
        return directional == {"turnon"}
    if action == "turnoff":
        return directional == {"turnoff"}
    if action == "set":
        return any(_phrase_in(normalized, term) for term in (
            "set", "change", "make", "move", "adjust", "open", "raise", "lower", "use", "confirm"
        ))
    if action in {"adjustup", "adjustdown"}:
        return any(_phrase_in(normalized, term) for term in (
            "raise", "lower", "increase", "decrease", "brighter", "dimmer", "warmer", "cooler", "adjust"
        ))
    return False


def _directional_actions(normalized_text: str) -> set[str]:
    """Extract action words without treating prepositions such as ``on Floor`` as actions."""

    result: set[str] = set()
    immediate = re.findall(r"\b(?:turn|switch)\s+(on|off)\b", normalized_text)
    result.update("turnon" if value == "on" else "turnoff" for value in immediate)
    # Support "switch that device off", but only when the direction ends the
    # clause.  This deliberately does not match "on the Ground Floor".  It is
    # collected even when another immediate action exists, so opposing clauses
    # cannot be silently collapsed into the first action.
    trailing = re.findall(
        r"\b(?:turn|switch)\s+(?:the\s+)?(?:that\s+)?(?:[a-z0-9]+\s+){0,8}"
        r"(on|off)(?=\s*(?:(?:right\s+)?now|please)?\s*(?:[,.!?;]|$))",
        normalized_text,
    )
    result.update("turnon" if value == "on" else "turnoff" for value in trailing)
    if _phrase_in(normalized_text, "open") or _phrase_in(normalized_text, "start"):
        result.add("turnon")
    if _phrase_in(normalized_text, "close") or _phrase_in(normalized_text, "shut"):
        result.add("turnoff")
    return result


def _unit_supported(text: str, instruction: DomuxInstruction) -> bool:
    unit = normalize_text(instruction.unit)
    if unit == "*":
        return True
    normalized = normalize_text(text)
    fahrenheit = _phrase_in(normalized, "fahrenheit") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*(?:°\s*)?f\b", normalized)
    )
    kelvin = _phrase_in(normalized, "kelvin") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*k\b", normalized)
    )
    celsius = _phrase_in(normalized, "celsius") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*(?:°\s*)?c\b", normalized)
    )
    if unit == "percent":
        return _phrase_in(normalized, "percent") or "%" in text or _phrase_in(normalized, "halfway")
    if unit == "celsius":
        if fahrenheit or kelvin:
            return False
        # In this English-only case, an otherwise unqualified "degree(s)" is
        # interpreted using the registered HA entity's Celsius capability.
        return celsius or any(_phrase_in(normalized, term) for term in ("degree", "degrees"))
    if unit == "kelvin":
        return kelvin and not (fahrenheit or celsius)
    if unit == "level":
        return any(_phrase_in(normalized, term) for term in ("level", "low", "medium", "high"))
    return False


def _distinct_named_values(text: str, values: Iterable[str]) -> frozenset[str]:
    """Return values with shorter matches removed only at overlapping spans."""

    normalized = normalize_text(text)
    matches: list[tuple[int, int, str]] = []
    for value in {normalize_text(item) for item in values if normalize_text(item)}:
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
            normalized,
        ):
            matches.append((match.start(), match.end(), value))
    kept = [
        item for item in matches
        if not any(
            other[:2] != item[:2]
            and other[0] <= item[0]
            and other[1] >= item[1]
            for other in matches
        )
    ]
    return frozenset(value for _start, _end, value in kept)


def _targeted_named_values(text: str, values: Iterable[str]) -> frozenset[str]:
    """Extract enum values that immediately follow an explicit target cue."""

    normalized = normalize_text(text)
    targets: set[str] = set()
    for value in _distinct_named_values(normalized, values):
        if re.search(
            rf"\b(?:to|into)\b(?:\s+(?:the|a|an|target|new)){{0,3}}\s+{re.escape(value)}\b",
            normalized,
        ):
            targets.add(value)
    return frozenset(targets)


def _instruction_numeric_value(instruction: DomuxInstruction) -> float | None:
    try:
        return float(instruction.value)
    except ValueError:
        return None


def _operational_conflicts(text: str, instruction: DomuxInstruction) -> frozenset[str]:
    """Find explicit semantics that one proposed seven-slot tuple cannot bind.

    Presence checks alone are unsafe for corrections such as ``from 50 to 20``:
    both numbers occur, but only 20 is the requested target.  This routine is a
    deliberately small fail-closed binder for directional actions, target
    numbers, colors, HVAC modes, and incompatible unit families.
    """

    normalized = normalize_text(text)
    conflicts: set[str] = set()
    if _has_operation_value_exclusion(text):
        # The seven-slot contract cannot encode "anything except X".  A
        # concrete, non-excluded replacement must be supplied in clarification.
        conflicts.add("value")
    directional_actions = _directional_actions(normalized)
    proposed_action = normalize_text(instruction.action)
    if len(directional_actions) > 1:
        conflicts.add("action")
    if "turnoff" in directional_actions and proposed_action in {"set", "adjustup", "adjustdown"}:
        # HA set/adjust services for these domains can turn a light/cover/AC on;
        # they cannot stand in for an explicit off/close operation.
        conflicts.add("action")

    numbers = set(_numbers_in(normalized))
    if _phrase_in(normalized, "halfway"):
        numbers.add(50.0)
    target_numbers = {
        float(value)
        for value in re.findall(
            r"\b(?:to|at|around|about)\s+(?:around\s+|about\s+)?(-?\d+(?:\.\d+)?)\b",
            normalized,
        )
    }
    by_numbers = {
        float(value)
        for value in re.findall(r"\bby\s+(-?\d+(?:\.\d+)?)\b", normalized)
    }
    relative_numbers = {
        float(value)
        for value in re.findall(
            r"(?<![a-z0-9])-?(\d+(?:\.\d+)?)\s*(?:%|percent|degrees?|celsius)?\s+"
            r"(?:brighter|dimmer|warmer|cooler|higher|lower)\b",
            normalized,
        )
    }
    if proposed_action in {"adjustup", "adjustdown"}:
        target_numbers.update(by_numbers | relative_numbers)
    if re.search(r"\b(?:to|at)\s+halfway\b", normalized):
        target_numbers.add(50.0)
    range_expression = bool(re.search(
        r"\bbetween\s+-?\d+(?:\.\d+)?\s+and\s+-?\d+(?:\.\d+)?\b|"
        r"\bfrom\s+-?\d+(?:\.\d+)?\s+(?:through|until)\s+-?\d+(?:\.\d+)?\b",
        normalized,
    ))
    inequality_expression = bool(re.search(
        r"(?:\b(?:below|under|above|over)\b|\b(?:less|greater)\s+than\b|[<>])"
        r"\s*-?\d+(?:\.\d+)?\b",
        normalized,
    ))
    numeric_from_to = bool(re.search(
        r"\bfrom\s+-?\d+(?:\.\d+)?\s+to\s+-?\d+(?:\.\d+)?\b",
        normalized,
    ))
    proposed_number = _instruction_numeric_value(instruction)
    if "turnon" in directional_actions and proposed_action in {"set", "adjustup", "adjustdown"}:
        attribute = normalize_text(instruction.attribute)
        compatible_turn_on = (
            attribute in {"brightness", "color", "colortemperature"}
            or (proposed_action == "set" and attribute == "mode")
            or (
                attribute in {"position", "openness"}
                and (
                    (proposed_action == "set" and proposed_number is not None and proposed_number > 0)
                    or proposed_action == "adjustup"
                )
            )
        )
        if not compatible_turn_on:
            conflicts.add("action")
    absolute_numeric_cue = bool(target_numbers - by_numbers - relative_numbers) or bool(
        re.search(r"\b(?:to|at)\s+halfway\b", normalized)
    )
    if proposed_action in {"adjustup", "adjustdown"} and absolute_numeric_cue:
        conflicts.update(("action", "value"))
    if proposed_action == "set" and by_numbers:
        conflicts.update(("action", "value"))
    if (
        proposed_action in {"adjustup", "adjustdown"}
        and numbers
        and not (by_numbers or relative_numbers)
    ):
        conflicts.add("value")
    if (
        range_expression
        or inequality_expression
        or len(target_numbers) > 1
        or (len(numbers) > 1 and not numeric_from_to)
    ):
        conflicts.add("value")
    elif len(target_numbers) == 1:
        target = next(iter(target_numbers))
        if proposed_number is None or not math.isclose(proposed_number, target, rel_tol=0, abs_tol=0.001):
            conflicts.add("value")
    elif len(numbers) == 1:
        only = next(iter(numbers))
        if proposed_number is None or not math.isclose(proposed_number, only, rel_tol=0, abs_tol=0.001):
            conflicts.add("value")

    colors = _distinct_named_values(normalized, COLOR_RGB)
    color_targets = _targeted_named_values(normalized, COLOR_RGB)
    proposed_value = normalize_text(instruction.value)
    color_from_to = bool(re.search(
        r"\bfrom\b(?:\s+[a-z0-9]+){0,3}\s+(?:" + "|".join(
            re.escape(color) for color in sorted(COLOR_RGB, key=len, reverse=True)
        ) + r")\s+to\s+(?:" + "|".join(
            re.escape(color) for color in sorted(COLOR_RGB, key=len, reverse=True)
        ) + r")\b",
        normalized,
    ))
    if len(color_targets) > 1 or (len(colors) > 1 and not color_from_to):
        conflicts.add("value")
    elif len(color_targets) == 1 and proposed_value != next(iter(color_targets)):
        conflicts.add("value")

    hvac_values = ("fan only", "heat cool", "cool", "heat", "dry", "fan", "auto")
    mode_text = re.sub(r"\b(?:fan|wind)\s+speed\b", " ", normalized)
    modes = _distinct_named_values(mode_text, hvac_values)
    mode_targets = _targeted_named_values(mode_text, hvac_values)

    def canonical_mode(value: str) -> str:
        return "fan only" if normalize_text(value) == "fan" else normalize_text(value)

    proposed_mode = canonical_mode(instruction.value)
    canonical_targets = {canonical_mode(value) for value in mode_targets}
    mode_pattern = r"(?:fan\s+only|heat\s+cool|cool|heat|dry|fan|auto)"
    mode_from_to = bool(re.search(
        rf"\bfrom\s+{mode_pattern}\s+to\s+{mode_pattern}\b",
        mode_text,
    ))
    if len(canonical_targets) > 1 or (len(modes) > 1 and not mode_from_to):
        conflicts.add("value")
    elif len(canonical_targets) == 1 and proposed_mode != next(iter(canonical_targets)):
        conflicts.add("value")

    # A mode/color plus a numeric target is more than one operation for this
    # seven-slot contract; do not let either proposed tuple silently drop half.
    if numbers and modes and any(_phrase_in(normalized, cue) for cue in ("ac", "air conditioner", "mode")):
        conflicts.update(("attribute", "value"))
    if numbers and colors and any(_phrase_in(normalized, cue) for cue in ("light", "lamp", "color")):
        conflicts.update(("attribute", "value"))

    families: set[str] = set()
    color_temperature_text = re.sub(r"\bcolor\s+temperature\b", " ", normalized)
    light_cue = any(_phrase_in(normalized, cue) for cue in ("light", "lamp"))
    cover_cue = any(_phrase_in(normalized, cue) for cue in ("curtain", "blind", "shade"))
    climate_cue = any(_phrase_in(normalized, cue) for cue in ("ac", "air conditioner", "air conditioning"))
    if any(_phrase_in(normalized, cue) for cue in ("brightness", "brighter", "dimmer")) or (
        light_cue and ("%" in text or _phrase_in(normalized, "percent"))
    ):
        families.add("brightness")
    if any(_phrase_in(normalized, cue) for cue in ("position", "openness")) or (
        cover_cue and (numbers or _phrase_in(normalized, "halfway"))
    ):
        families.add("position")
    if _phrase_in(normalized, "color temperature") or _phrase_in(normalized, "kelvin") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*k\b", normalized)
    ):
        families.add("color_temperature")
    if colors or _phrase_in(color_temperature_text, "color"):
        families.add("color")
    temperature_text = re.sub(r"\bcolor\s+temperature\b", " ", normalized)
    if any(_phrase_in(temperature_text, cue) for cue in (
        "temperature", "celsius", "fahrenheit", "degree", "degrees",
    )) or (
        climate_cue and (numbers or any(_phrase_in(normalized, cue) for cue in ("warmer", "cooler")))
    ):
        families.add("temperature")
    if _phrase_in(normalized, "mode") or (climate_cue and modes):
        families.add("mode")
    if any(_phrase_in(normalized, cue) for cue in ("wind speed", "fan speed")):
        families.add("fan_speed")
    if len(families) > 1:
        conflicts.add("attribute")
    setter_cue = any(_phrase_in(normalized, cue) for cue in (
        "set", "change", "make", "adjust", "raise", "lower", "use",
    ))
    if proposed_action in {"turnon", "turnoff"} and setter_cue and families:
        conflicts.update(("action", "attribute"))

    fahrenheit = _phrase_in(normalized, "fahrenheit") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*(?:°\s*)?f\b", normalized)
    )
    kelvin = _phrase_in(normalized, "kelvin") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*k\b", normalized)
    )
    celsius = _phrase_in(normalized, "celsius") or bool(
        re.search(r"-?\d+(?:\.\d+)?\s*(?:°\s*)?c\b", normalized)
    )
    proposed_unit = normalize_text(instruction.unit)
    if (
        (proposed_unit == "celsius" and (fahrenheit or kelvin))
        or (proposed_unit == "kelvin" and (fahrenheit or celsius))
        or (proposed_unit not in {"*", "celsius", "kelvin"} and fahrenheit)
    ):
        conflicts.add("unit")
    return frozenset(conflicts)


def _source_selector(utterance: str, instruction: DomuxInstruction) -> DomuxInstruction:
    """Erase model-proposed grounding fields that the user's words do not support."""

    normalized = normalize_text(utterance)
    fields = {slot: getattr(instruction, slot) for slot in SLOTS}
    if not _phrase_in(normalized, instruction.device):
        fields["device"] = "*"
    if not _phrase_in(normalized, instruction.room):
        fields["room"] = "*"
    if not _phrase_in(normalized, instruction.floor):
        fields["floor"] = "*"
    if not _attribute_supported(utterance, instruction):
        fields["attribute"] = "*"
    return DomuxInstruction(**fields)


def _missing_required_slots(instruction: DomuxInstruction) -> tuple[str, ...]:
    action = normalize_text(instruction.action)
    missing: list[str] = []
    if action in {"set", "adjustup", "adjustdown"} and instruction.attribute == "*":
        missing.append("attribute")
    if action == "set" and instruction.value == "*":
        missing.append("value")
    if action == "set" and instruction.attribute not in {"*", "color", "mode"} and instruction.unit == "*":
        missing.append("unit")
    return tuple(missing)


def _has_uncertainty_or_conflict(utterance: str) -> bool:
    normalized = normalize_text(utterance)
    return "—" in utterance or any(_phrase_in(normalized, cue) for cue in (
        "perhaps", "maybe", "confirm", "which", "not sure", "not decided",
        "have not decided", "ask me", "do not choose", "do not guess", "or did",
    )) or _phrase_in(normalized, "or") or _phrase_in(normalized, "either") or bool(
        re.search(r"\b(?:not|except|instead\s+of)\s+(?:the\s+)?[a-z0-9]", normalized)
    )


def _has_negative_action_authorization(text: str) -> bool:
    """Detect negation scoped over execution, even with intervening pronouns."""

    normalized = normalize_text(text)
    action = (
        r"(?:turn|switch|open|close|set|change|make|adjust|raise|lower|execute|"
        r"proceed|act|go\s+ahead|do\s+it|touch|confirm|authorize|approve|dispatch)"
    )
    bridge = r"(?:(?:you|me|us|i|we|to|need|want|let|allow|please|just)\s+){0,8}"
    return bool(re.search(
        rf"\b(?:do\s+not|don't|dont|never(?:\s+ever)?|rather\s+not)\s+{bridge}{action}\b|"
        rf"\b(?:must|may|shall|should|can|could|will|would)\s+not\s+{bridge}{action}\b|"
        rf"\bnot\s+(?:to\s+)?{action}\b|"
        rf"\b(?:refuse|forbid)\b.{{0,64}}\b{action}\b|"
        rf"\b(?:withdraw|revoke|deny)\b.{{0,40}}\b(?:permission|authorization|consent)\b"
        rf"(?:.{{0,64}}\b{action}\b)?|"
        rf"\bunder\s+no\s+circumstances\b.{{0,64}}\b{action}\b",
        normalized,
    ))


def _has_negative_or_cancelled_intent(utterance: str) -> bool:
    normalized = normalize_text(utterance)
    negative_imperative = _has_negative_action_authorization(normalized) or bool(re.search(
        r"\b(?:do\s+not|don't|dont|never(?:\s+ever)?)\s+"
        r"(?:turn|switch|open|close|set|change|make|adjust|raise|lower)\b",
        normalized,
    ))
    withdrawn_request = bool(re.search(
        r"\b(?:i\s+)?(?:do\s+not|don't)\s+(?:want|need)\b.{0,48}\b"
        r"(?:turn|switch|open|close|set|change|make|adjust|raise|lower)\b|"
        r"\bno\s+need\s+to\s+(?:turn|switch|open|close|set|change|make|adjust|raise|lower)\b|"
        r"\b(?:refrain\s+from|avoid)\s+(?:turning|switching|opening|closing|setting|"
        r"changing|making|adjusting|raising|lowering)\b|"
        r"\bwithout\s+(?:turning|switching|opening|closing|setting|changing|making|"
        r"adjusting|raising|lowering)\b|"
        r"\bjust\s+kidding\b|"
        r"\b(?:scratch|nix|disregard|ignore)\s+(?:it|that|this)\b|"
        r"\b(?:i\s+)?changed?\s+my\s+mind\b|\bhold\s+on\b|"
        r"\b(?:actually\s+)?(?:never\s+mind|cancel(?:\s+(?:it|that|this|the\s+request))?)\b",
        normalized,
    ))
    terminal_cancel = bool(re.fullmatch(
        r"(?:cancel|never mind|do nothing|stop|wait|forget it|"
        r"(?:actually\s+)?(?:do not|don't|dont))[.!]?",
        normalized,
    ))
    trailing_cancel = bool(re.search(
        r"(?:^|[,;:—-]\s*)(?:(?:(?:please|i\s+mean)\s+)?no(?:\s+please)?|"
        r"i\s+want\s+no|not now|forget it|wait|stop|don't bother|dont bother|"
        r"(?:i\s+)?(?:do not|don't|dont)\s+(?:want|need)\s+(?:it|that|this)|"
        r"(?:actually\s+|please\s+)?(?:do not|don't|dont))[.!]?\s*$",
        normalized,
    ))
    return negative_imperative or withdrawn_request or terminal_cancel or trailing_cancel


def _has_unsupported_condition_or_time(utterance: str) -> bool:
    """Reject modifiers this immediate, single-action executor cannot honor."""

    normalized = normalize_text(utterance)
    # "Confirm ... before acting" describes this safety protocol itself, not
    # a delayed execution condition.  Keep it available to clarification.
    normalized = re.sub(
        r"\bbefore\s+(?:acting|executing|proceeding|you\s+act|you\s+execute)\b",
        " ",
        normalized,
    )
    number_word = (
        r"(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"few|couple(?:\s+of)?|\d+(?:\.\d+)?)"
    )
    return bool(re.search(
        r"\b(?:if|unless|when|whenever|once|while|after|before|until|provided|providing|assuming)\b|"
        r"\b(?:as\s+long\s+as|in\s+case)\b|"
        r"\bas\s+soon\s+as\b|"
        r"\b(?:tomorrow|tonight|later|today|noon|midnight|sunrise|sunset|"
        r"morning|afternoon|evening)\b|"
        r"\b(?:this|next)\s+(?:morning|afternoon|evening|night|week|month|year)\b|"
        r"\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"weekday|weekend)\b|"
        r"\b(?:every|each)\s+(?:morning|afternoon|evening|night|day|week|month|"
        r"weekday|weekend)\b|"
        r"\b(?:daily|nightly|weekly|monthly|briefly|temporarily|schedule|scheduled)\b|"
        rf"\b(?:in|for)\s+{number_word}\s+(?:seconds?|minutes?|hours?|days?|weeks?)\b|"
        r"\b(?:in|for)\s+(?:half|a\s+half|quarter|a\s+quarter)\s+"
        r"(?:of\s+)?(?:an?\s+)?(?:hour|day)\b|"
        r"\bat\s+(?:noon|midnight|sunrise|sunset)\b|"
        r"\bat\s+\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b|"
        r"\bat\s+\d{1,2}\s*(?:am|pm)\b|"
        r"\bat\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+o'?clock)?\b",
        normalized,
    ))


def _has_informational_request(utterance: str) -> bool:
    """Distinguish questions about an action from an authorization to execute it."""

    normalized = re.sub(r"^[^a-z0-9]+", "", normalize_text(utterance))
    return bool(re.match(
        r"^(?:(?:should|would|could|can|may)\s+i\b|"
        r"(?:would|could)\s+it\s+be\s+(?:safe|okay|ok|wise|advisable)\b|"
        r"is\s+it\s+(?:safe|okay|ok|wise|advisable)\b|"
        r"do\s+you\s+(?:recommend|suggest|think)\b|"
        r"do\s+i\s+(?:need|have)\s+to\b|"
        r"are\s+you\s+going\s+to\b|"
        r"why\s+(?:should|would|could|can|do)\s+i\b|"
        r"(?:please\s+)?explain\s+why\b|"
        r"what\s+if\b|"
        r"what\s+(?:happens|would\s+happen)\b|"
        r"(?:tell|show|explain)\s+me\s+(?:how|whether|what)\b|"
        r"how\s+(?:do|can|should|would)\s+i\b)",
        normalized,
    ))


def _has_supported_request_grammar(
    utterance: str,
    registry: EntityRegistry,
    source_instructions: Sequence[DomuxInstruction] = (),
) -> bool:
    """Accept only the documented immediate, single-operation language.

    Negative and temporal detectors provide useful reason codes, but they can
    never enumerate every English paraphrase.  This bounded vocabulary is the
    fail-closed backstop: registry labels, numeric values, and the small Domux
    operation language are accepted; an unconsumed word requires a new request
    rather than being silently discarded from an executable plan.
    """

    selector_words: set[str] = set()
    for entity in registry.entities:
        for label in (
            entity.entity_id, entity.room, entity.floor, entity.device, *entity.aliases,
        ):
            selector_words.update(re.findall(r"[a-z0-9]+", normalize_text(label)))
    for alias in GENERIC_DEVICE_ALIASES:
        selector_words.update(re.findall(r"[a-z0-9]+", normalize_text(alias)))
    # Permit an unknown single-token mode only when the user's own syntax binds
    # it as ``to <value> mode`` and the model puts that exact token in value.
    # Attribute/unit text never extends the grammar: otherwise an untrusted raw
    # output could launder an ignored condition into a replaceable source slot.
    normalized_utterance = normalize_text(utterance)
    source_mode_words = {
        value
        for instruction in source_instructions
        if normalize_text(instruction.attribute) == "mode"
        if re.fullmatch(r"[a-z0-9]+", (value := normalize_text(instruction.value)))
        if re.search(
            rf"\bto\s+{re.escape(value)}\s+mode\b",
            normalized_utterance,
        )
    }

    request_words = {
        # Immediate command and clarification-protocol framing.
        "adjust", "ask", "acting", "avoid", "before", "change", "check", "choose",
        "close", "confirm", "decided", "decrease", "did", "do", "execute",
        "guess", "increase", "lower", "make", "mean", "move", "need", "open",
        "proceed", "raise", "set", "switch", "then", "turn", "value", "wait", "want",
        # Operation attributes, values, and units.
        "auto", "blue", "bright", "brighter", "brightness", "celsius", "color",
        "cool", "cooler", "degree", "degrees", "dimmer", "dry", "fan", "green",
        "fahrenheit", "halfway", "heat", "high", "kelvin", "level", "low", "medium", "mode",
        "off", "on", "openness", "percent", "position", "red", "speed",
        "temperature", "warm", "warmer", "white", "wind", "yellow",
        # Selector and polite-command glue.  These words carry no operation by
        # themselves; the source-to-slot checks still require positive evidence.
        "about", "ac", "air", "am", "and", "around", "at", "balcony", "bedroom",
        "any", "anything", "apart", "besides", "but", "by", "can", "conditioner",
        "could", "curtain", "device", "downstairs", "east",
        "floor", "for", "from", "have", "i", "in", "instead", "it", "its", "just",
        "lab", "light", "main", "me", "middle", "my", "no", "not", "now", "of",
        "office", "one", "or", "other", "perhaps", "please", "reading", "right", "room",
        "something", "than",
        "side", "studio", "sure", "talked", "that", "the", "this", "to", "upstairs",
        "use", "we", "west", "which", "would", "you",
        *{word for color in COLOR_RGB for word in normalize_text(color).split()},
    }
    allowed = selector_words | request_words | source_mode_words
    normalized = normalized_utterance.replace("don't", "do not").replace("dont", "do not")
    words = re.findall(r"[a-z0-9]+", normalized)
    return all(
        word in allowed or word.isdigit() or re.fullmatch(r"\d+(?:\d+)?k", word)
        for word in words
    )


def _has_unresolved_generic_exclusion(text: str) -> bool:
    """Reject deictic exclusions that do not identify a stable entity ID."""

    return bool(re.search(
        r"\bnot\s+(?:(?:this|that|the)\s+)?(?:one|device)\b|"
        r"\bnot\s+(?:this|that)\b|"
        r"\b(?:leave|keep)\s+(?:(?:this|that|the)\s+)?(?:one|device)\s+"
        r"(?:unchanged|as\s+is)\b|"
        r"\b(?:use|select|choose|mean)\s+(?:the\s+)?other\s+"
        r"(?:one|device|light|lamp|curtain|blind|shade|ac|air\s+conditioner)\b",
        normalize_text(text),
    ))


def _explicit_operational_requirements(utterance: str) -> frozenset[str]:
    """Return operational slots explicitly present in the user's own words.

    This is a fail-closed preservation check, not a replacement semantic
    parser.  Its purpose is to stop a model proposal from silently dropping a
    number, unit, or attribute that would materially change the operation.
    """

    normalized = normalize_text(utterance)
    requirements: set[str] = set()
    has_operational_number = bool(re.search(
        r"\b(?:to|by|at|around|about)\s+-?\d+(?:\.\d+)?\b|"
        r"\b(?:brightness|position|openness|temperature|degrees?|celsius|kelvin)\b"
        r"(?:\s+[a-z]+){0,3}\s+-?\d+(?:\.\d+)?\b",
        normalized,
    )) or _phrase_in(normalized, "halfway")
    percent = "%" in utterance or _phrase_in(normalized, "percent") or _phrase_in(normalized, "halfway")
    celsius = any(_phrase_in(normalized, term) for term in ("celsius", "degree", "degrees"))
    kelvin = _phrase_in(normalized, "kelvin") or bool(re.search(r"\d+(?:\.\d+)?\s*k\b", normalized))
    if has_operational_number:
        requirements.add("value")
    if percent or celsius or kelvin:
        requirements.update(("attribute", "unit", "value"))
    if any(_phrase_in(normalized, term) for term in (
        "brightness", "bright", "dimmer", "position", "openness", "temperature",
        "color temperature", "wind speed", "fan speed",
    )):
        requirements.add("attribute")
    named_color = any(_phrase_in(normalized, color) for color in COLOR_RGB)
    light_cue = any(_phrase_in(normalized, term) for term in ("light", "lamp"))
    color_operation = any(_phrase_in(normalized, term) for term in ("make", "color")) or bool(
        re.search(r"\bto\b(?:\s+[a-z0-9]+){0,4}\s+(?:" + "|".join(
            re.escape(normalize_text(color)) for color in COLOR_RGB
        ) + r")\b", normalized)
    )
    if named_color and light_cue and color_operation:
        requirements.update(("attribute", "value"))
    mode_cue = _phrase_in(normalized, "mode")
    named_mode = any(_phrase_in(normalized, mode) for mode in ("cool", "heat", "dry", "fan only", "auto"))
    climate_cue = any(_phrase_in(normalized, term) for term in ("ac", "air conditioner", "air conditioning"))
    mode_operation = mode_cue or bool(re.search(
        r"\b(?:to|use)\b(?:\s+[a-z0-9]+){0,4}\s+(?:cool|heat|dry|fan\s+only|auto)\b",
        normalized,
    ))
    if mode_cue or (named_mode and climate_cue and mode_operation):
        requirements.add("attribute")
    if named_mode and climate_cue and mode_operation:
        requirements.add("value")
    return frozenset(requirements)


def _label_is_excluded(text: str, label: str) -> bool:
    normalized_label = normalize_text(label)
    if not normalized_label:
        return False
    return bool(re.search(
        rf"\b(?:not(?:\s+in)?|except|besides|other\s+than|anything\s+but|"
        rf"apart\s+from|instead\s+of)\s+(?:the\s+)?{re.escape(normalized_label)}\b|"
        rf"\bleave\s+(?:the\s+)?{re.escape(normalized_label)}\b.*\bunchanged\b",
        normalize_text(text),
    ))


def _negated_entity_ids(registry: EntityRegistry, utterance: str) -> tuple[str, ...]:
    normalized = normalize_text(utterance)
    excluded: list[str] = []
    for entity in registry.entities:
        labels = (entity.room, entity.floor, entity.device, *entity.aliases)
        if any(
            _label_is_excluded(normalized, label)
            for label in labels
            if normalize_text(label) not in {"light", "curtain", "ac"}
        ):
            excluded.append(entity.entity_id)
    return tuple(sorted(set(excluded)))


def _operational_text(text: str, entities: Sequence[EntitySpec]) -> str:
    """Remove registry selector spans before interpreting values or attributes."""

    result = normalize_text(text)
    labels: set[str] = set()
    for entity in entities:
        values = [entity.room, entity.floor, *entity.aliases]
        if normalize_text(entity.device) not in GENERIC_DEVICE_ALIASES:
            values.append(entity.device)
        labels.update(normalize_text(value) for value in values if normalize_text(value))
    for label in sorted(labels, key=len, reverse=True):
        result = re.sub(
            rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])",
            " ",
            result,
        )
    return " ".join(result.split())


def _clarification_operational_text(answer: str, chosen: EntitySpec) -> str:
    """Remove candidate-selection evidence before interpreting operation slots."""

    result = normalize_text(answer)
    generic_device_labels = tuple(
        alias for alias, domain in GENERIC_DEVICE_ALIASES.items()
        if domain == chosen.domain
    )
    labels = (
        chosen.entity_id,
        chosen.room,
        chosen.floor,
        chosen.device,
        *chosen.aliases,
        *generic_device_labels,
    )
    for label in sorted({normalize_text(value) for value in labels if normalize_text(value)}, key=len, reverse=True):
        result = re.sub(
            rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])",
            " ",
            result,
        )
    return " ".join(result.split())


def _answer_operational_requirements(answer: str, chosen: EntitySpec) -> frozenset[str]:
    operation_only = _clarification_operational_text(answer, chosen)
    requirements = set(_explicit_operational_requirements(operation_only))
    if chosen.domain == "light" and any(_phrase_in(operation_only, color) for color in COLOR_RGB):
        requirements.update(("attribute", "value"))
    if chosen.domain == "climate" and any(
        _phrase_in(operation_only, mode) for mode in ("cool", "heat", "dry", "fan only", "auto")
    ):
        requirements.update(("attribute", "value"))
    return frozenset(requirements)


def _clarification_has_positive_authorization(
    answer: str,
    chosen: EntitySpec,
    candidates: Sequence[EntitySpec] = (),
) -> bool:
    """Recognize a bounded fail-closed clarification-answer grammar.

    Candidate metadata and a small positive operation vocabulary are allowed;
    any unparsed residual word rejects the answer.  Merely finding an action
    token somewhere in arbitrary prose is never sufficient authorization.
    """

    answer_normalized = normalize_text(answer)
    if _has_negative_action_authorization(answer_normalized):
        return False
    if answer_normalized.isdigit():
        return True
    operation_only = _clarification_operational_text(answer, chosen)
    ordered_words = re.findall(r"[a-z0-9]+", operation_only)
    words = set(ordered_words)
    selector_fillers = {
        "the", "one", "option", "number", "device", "please", "room", "floor",
        "in", "at", "on", "located", "by", "other", "i", "mean", "and", "for",
        "to", "use", "not", "leave", "unchanged",
    }
    if not words or words.issubset(selector_fillers):
        return True
    selector_words: set[str] = set()
    for entity in (*candidates, chosen):
        for label in (
            entity.entity_id, entity.room, entity.floor, entity.device, *entity.aliases,
        ):
            selector_words.update(re.findall(r"[a-z0-9]+", normalize_text(label)))
    for alias in GENERIC_DEVICE_ALIASES:
        selector_words.update(re.findall(r"[a-z0-9]+", normalize_text(alias)))
    operation_words = {
        "yes", "confirm", "confirmed", "proceed", "go", "ahead", "do", "execute",
        "it", "that", "this", "turn", "switch", "open", "close", "shut", "start",
        "set", "change", "make", "move", "adjust", "raise", "lower", "increase",
        "decrease", "brighter", "dimmer", "warmer", "cooler", "temperature",
        "brightness", "position", "openness", "color", "mode", "fan", "wind",
        "speed", "celsius", "kelvin", "percent", "degree", "degrees", "halfway",
        "instead", "its", "right", "now", "off", "low", "medium", "high", "level",
        "cool", "heat", "dry", "auto", "only", "from", "into", "target", "new",
        *{word for color in COLOR_RGB for word in normalize_text(color).split()},
    }
    allowed = selector_fillers | selector_words | operation_words
    if any(word not in allowed and not word.isdigit() for word in ordered_words):
        return False
    affirmative = bool(re.fullmatch(
        r"(?:please\s+)?(?:yes(?:\s+please)?|confirm(?:ed)?|proceed|go\s+ahead|"
        r"do\s+it(?:\s+(?:right\s+)?now)?|execute\s+it)[.!]?",
        operation_only.strip(),
    ))
    identifies_candidate = any(
        _phrase_in(answer_normalized, label)
        for label in (chosen.entity_id, chosen.room, chosen.floor, chosen.device, *chosen.aliases)
    )
    requirements = _answer_operational_requirements(answer, chosen)
    directions = _directional_actions(operation_only)
    return affirmative or bool(requirements) or bool(directions) or identifies_candidate


def _mentioned_entities(
    registry: EntityRegistry,
    utterance: str,
    source_instructions: Sequence[DomuxInstruction],
) -> tuple[EntitySpec, ...]:
    normalized = normalize_text(utterance)
    hinted_domains = {
        domain for instruction in source_instructions
        if (domain := EntityRegistry._domain_hint(instruction)) is not None
    }
    mentioned: list[EntitySpec] = []
    for entity in registry.entities:
        if hinted_domains and entity.domain not in hinted_domains:
            continue
        discriminators = (entity.room, *entity.aliases)
        if any(_phrase_in(normalized, value) for value in discriminators):
            mentioned.append(entity)
    return tuple(mentioned)


def _explicit_mentioned_entities(
    registry: EntityRegistry,
    utterance: str,
) -> tuple[EntitySpec, ...]:
    """Resolve every explicitly named domain/room pair, independent of model output."""

    normalized = normalize_text(utterance)
    mentioned_domains = {
        domain
        for alias, domain in GENERIC_DEVICE_ALIASES.items()
        if _phrase_in(normalized, alias)
    }
    mentioned_rooms = set(_distinct_named_values(
        normalized, (entity.room for entity in registry.entities),
    ))
    mentioned_floors = set(_distinct_named_values(
        normalized, (entity.floor for entity in registry.entities),
    ))
    mentioned_aliases = set(_distinct_named_values(
        normalized,
        (alias for entity in registry.entities for alias in entity.aliases),
    ))
    selector_present = bool(mentioned_rooms or mentioned_floors or mentioned_aliases)
    specific_device_ids = {
        entity.entity_id
        for entity in registry.entities
        if normalize_text(entity.device) not in GENERIC_DEVICE_ALIASES
        and _phrase_in(normalized, entity.device)
    }
    mentioned: list[EntitySpec] = []
    for entity in registry.entities:
        room_match = normalize_text(entity.room) in mentioned_rooms
        floor_match = normalize_text(entity.floor) in mentioned_floors
        alias_match = any(normalize_text(alias) in mentioned_aliases for alias in entity.aliases)
        # A single named room and floor jointly identify an entity.  When the
        # utterance names several rooms/floors, keep every matching target so a
        # one-tuple model output cannot silently truncate the request.
        discriminator = (
            (not mentioned_rooms or room_match)
            and (not mentioned_floors or floor_match)
        ) or alias_match
        exact_alias = any(_phrase_in(normalized, alias) for alias in entity.aliases)
        device_match = (
            entity.entity_id in specific_device_ids
            if specific_device_ids
            else entity.domain in mentioned_domains
        )
        if (not selector_present or discriminator) and (device_match or exact_alias):
            mentioned.append(entity)
    return tuple(mentioned)


def _request_candidates(
    utterance: str,
    source_instructions: Sequence[DomuxInstruction],
    registry: EntityRegistry,
    context: SessionContext,
) -> tuple[tuple[DomuxInstruction, ...], tuple[EntitySpec, ...]]:
    selectors = tuple(_source_selector(utterance, instruction) for instruction in source_instructions)
    by_id: dict[str, EntitySpec] = {}
    for selector in selectors:
        for entity in registry.candidates(selector, context):
            by_id[entity.entity_id] = entity
    if _has_uncertainty_or_conflict(utterance):
        for entity in _mentioned_entities(registry, utterance, source_instructions):
            by_id[entity.entity_id] = entity
    explicitly_mentioned = _explicit_mentioned_entities(registry, utterance)
    if len(explicitly_mentioned) > 1:
        for entity in explicitly_mentioned:
            by_id[entity.entity_id] = entity
    return selectors, tuple(sorted(by_id.values(), key=registry._sort_key))


@dataclass(frozen=True)
class GroundedRequest:
    utterance: str
    raw_output: str
    source_instructions: tuple[DomuxInstruction, ...]
    selector_instructions: tuple[DomuxInstruction, ...]
    context_entity_ids: tuple[str, ...]
    negated_entity_ids: tuple[str, ...]
    excluded_operation_value_tokens: tuple[str, ...]
    candidates: tuple[EntitySpec, ...]
    clarification: Clarification
    request_digest: str


@dataclass(frozen=True)
class ResolvedRequest:
    grounded: GroundedRequest
    chosen: EntitySpec
    confirmed_instruction: DomuxInstruction
    clarification_digest: str


def ground_domux_request(
    utterance: str,
    raw_output: str,
    registry: EntityRegistry,
    context: SessionContext | None = None,
) -> GroundedRequest:
    if not isinstance(utterance, str) or not utterance.strip():
        raise GroundingError("user utterance is empty")
    context = context or SessionContext()
    source = parse_domux_output(raw_output)
    selectors, candidates = _request_candidates(utterance, source, registry, context)
    operational_utterance = _operational_text(utterance, candidates or registry.entities)
    negated_entity_ids = _negated_entity_ids(registry, utterance)
    excluded_value_tokens = _excluded_operation_value_tokens(utterance, source)
    reasons: list[str] = []
    unresolved: list[str] = []
    if not candidates:
        reasons.append("no_registry_match")
    elif len(candidates) > 1:
        reasons.append("multiple_registry_matches")
    if len(source) != 1:
        reasons.append("multiple_model_instructions")
    if _has_uncertainty_or_conflict(utterance):
        reasons.append("uncertainty_or_conflict")
    if _has_negative_or_cancelled_intent(utterance):
        reasons.append("negative_or_cancelled_intent")
    if not _has_supported_request_grammar(utterance, registry, source):
        reasons.append("unsupported_request_grammar")
        unresolved.append("authorization")
    if _phrase_in(normalize_text(utterance), "other one") and not context.recent_entity_ids:
        reasons.append("unsupported_request_grammar")
        unresolved.append("authorization")
    if _has_unresolved_generic_exclusion(utterance):
        reasons.append("unsupported_request_grammar")
        unresolved.append("authorization")
    informational_request = _has_informational_request(utterance)
    if informational_request:
        reasons.append("informational_request")
        unresolved.append("authorization")
    elif _has_unsupported_condition_or_time(utterance):
        reasons.append("unsupported_condition_or_time")
        unresolved.append("condition_or_time")
    if negated_entity_ids:
        reasons.append("negated_selector")
    if _has_operation_value_exclusion(operational_utterance):
        reasons.append("excluded_operation_value")
        unresolved.append("value")
    explicit_requirements = _explicit_operational_requirements(operational_utterance)
    for instruction, selector in zip(source, selectors):
        for slot in ("device", "attribute", "room", "floor"):
            if getattr(instruction, slot) != getattr(selector, slot):
                unresolved.append(slot)
        unresolved.extend(_missing_required_slots(instruction))
        if not _action_supported(operational_utterance, instruction):
            unresolved.append("action")
        if not _attribute_supported(operational_utterance, instruction) and not (
            len(candidates) == 1
            and _attribute_supported_for_entity(operational_utterance, instruction, candidates[0])
        ):
            unresolved.append("attribute")
        if not _value_supported(operational_utterance, instruction.value):
            unresolved.append("value")
        if not _unit_supported(operational_utterance, instruction):
            unresolved.append("unit")
        unresolved.extend(sorted(_operational_conflicts(operational_utterance, instruction)))
        if normalize_text(instruction.action) == "turnon" and any(
            candidate.domain == "climate" for candidate in candidates
        ):
            reasons.append("climate_mode_confirmation_required")
            unresolved.extend(("attribute", "value"))
        for slot in sorted(explicit_requirements):
            if getattr(instruction, slot) == "*":
                unresolved.append(slot)
    if unresolved:
        reasons.append("ungrounded_or_missing_slots")
    reasons = list(dict.fromkeys(reasons))
    unresolved_slots = tuple(dict.fromkeys(unresolved))

    base = clarification_for(candidates)
    required = bool(reasons)
    if not required:
        clarification = base
    elif not candidates or any(reason in reasons for reason in (
        "negative_or_cancelled_intent", "informational_request", "unsupported_condition_or_time",
        "unsupported_request_grammar",
    )):
        clarification = Clarification(
            True,
            next((reason for reason in (
                "negative_or_cancelled_intent", "informational_request", "unsupported_condition_or_time",
                "unsupported_request_grammar",
            ) if reason in reasons), reasons[0]),
            tuple(candidates[:3]),
            None,
            tuple(reasons),
            unresolved_slots,
        )
    elif len(candidates) > 3:
        displayed = tuple(candidates[:3])
        clarification = Clarification(
            True,
            "too_many_candidates",
            displayed,
            "More than three devices match. Please provide a room or floor before choosing.",
            tuple(dict.fromkeys((*reasons, "too_many_candidates"))),
            unresolved_slots,
        )
    else:
        displayed = tuple(candidates[:3])
        options = "; ".join(
            _candidate_option(index, candidate)
            for index, candidate in enumerate(displayed, start=1)
        )
        detail = ", ".join(unresolved_slots) if unresolved_slots else "device or value"
        clarification = Clarification(
            True,
            reasons[0],
            displayed,
            f"Please confirm {detail}. Candidates: {options}.",
            tuple(reasons),
            unresolved_slots,
        )
    context_ids = tuple(dict.fromkeys(context.recent_entity_ids))
    payload = {
        "utterance_sha256": hashlib.sha256(utterance.encode("utf-8")).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "source": [instruction.to_pipe() for instruction in source],
        "selectors": [instruction.to_pipe() for instruction in selectors],
        "context_entity_ids": context_ids,
        "negated_entity_ids": negated_entity_ids,
        "excluded_operation_value_tokens": excluded_value_tokens,
        "candidate_ids": [candidate.entity_id for candidate in candidates],
        "reasons": clarification.reasons,
        "unresolved_slots": clarification.unresolved_slots,
    }
    return GroundedRequest(
        utterance=utterance,
        raw_output=raw_output,
        source_instructions=source,
        selector_instructions=selectors,
        context_entity_ids=context_ids,
        negated_entity_ids=negated_entity_ids,
        excluded_operation_value_tokens=excluded_value_tokens,
        candidates=candidates,
        clarification=clarification,
        request_digest=digest_json(payload),
    )


def resolve_clarification(answer: str, candidates: Sequence[EntitySpec]) -> EntitySpec:
    if not isinstance(answer, str) or not answer.strip():
        raise GroundingError("clarification answer is empty")
    answer_norm = normalize_text(answer)
    if _has_unresolved_generic_exclusion(answer_norm):
        raise GroundingError("clarification answer uses an unresolved generic exclusion")
    other = re.search(r"\b(?:the\s+)?other\s+one\b", answer_norm)
    if other is not None:
        suffix = answer_norm[other.end():]
        explicit_after = any(
            _phrase_in(suffix, label)
            for entity in candidates
            for label in (entity.entity_id, entity.room, entity.floor, *entity.aliases)
        )
        if not explicit_after:
            raise GroundingError("clarification answer does not identify which other candidate")
    if answer_norm.isdigit():
        index = int(answer_norm) - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        raise GroundingError("clarification index is outside the displayed candidates")
    exact_id = [entity for entity in candidates if normalize_text(entity.entity_id) == answer_norm]
    if len(exact_id) == 1:
        return exact_id[0]
    if len(candidates) == 1:
        candidate = candidates[0]
        for label in (candidate.room, candidate.floor, candidate.device, *candidate.aliases):
            if _label_is_excluded(answer_norm, label):
                raise GroundingError("clarification answer excludes the only candidate")
        if _answer_is_noncommittal(answer_norm):
            raise GroundingError("clarification answer is noncommittal")
        identifies_candidate = any(
            _phrase_in(answer_norm, label)
            for label in (candidate.entity_id, candidate.room, candidate.floor, *candidate.aliases)
        )
        supplies_operation = bool(_explicit_operational_requirements(answer)) or bool(
            _directional_actions(answer_norm)
        ) or bool(_numbers_in(answer_norm)) or any(
            _phrase_in(answer_norm, value)
            for value in (*COLOR_RGB, "cool", "heat", "dry", "fan only", "auto", "low", "medium", "high")
        )
        explicitly_affirms = bool(re.match(
            r"^(?:yes\b|confirm\b|confirmed\b|proceed\b|do\s+it\b|that\s+one\b)",
            answer_norm,
        ))
        if not (identifies_candidate or supplies_operation or explicitly_affirms):
            raise GroundingError("clarification answer provides no confirmation evidence")
        return candidate
    feature_counts: dict[tuple[str, str], int] = {}
    for entity in candidates:
        for kind, value in (("room", entity.room), ("floor", entity.floor), ("device", entity.device)):
            key = (kind, normalize_text(value))
            feature_counts[key] = feature_counts.get(key, 0) + 1
    scored: list[tuple[int, EntitySpec]] = []
    for entity in candidates:
        negative = False
        for label in (entity.room, entity.floor, entity.device, *entity.aliases):
            if _label_is_excluded(answer_norm, label):
                negative = True
        if negative:
            continue
        score = 0
        combined = normalize_text(f"{entity.floor} {entity.room} {entity.device}")
        if _phrase_in(answer_norm, combined):
            score += 20
        for kind, value, weight in (
            ("room", entity.room, 8), ("floor", entity.floor, 5), ("device", entity.device, 4)
        ):
            key = (kind, normalize_text(value))
            if feature_counts[key] < len(candidates) and _phrase_in(answer_norm, value):
                score += weight
        score += 10 * sum(_phrase_in(answer_norm, alias) for alias in entity.aliases)
        if score:
            scored.append((score, entity))
    if not scored:
        raise GroundingError("clarification answer does not select a candidate")
    best = max(score for score, _ in scored)
    matches = [entity for score, entity in scored if score == best]
    if len(matches) != 1:
        raise GroundingError(f"clarification answer selects {len(matches)} candidates")
    return matches[0]


def _validate_confirmed_instruction(
    grounded: GroundedRequest,
    answer: str,
    confirmed: DomuxInstruction,
    chosen: EntitySpec,
    registry: EntityRegistry,
) -> None:
    answer_normalized = normalize_text(answer)
    if _answer_cancels(answer_normalized) or _has_negative_or_cancelled_intent(answer) or any(
        _phrase_in(answer_normalized, phrase)
        for phrase in ("do not act", "do not execute", "don't act", "don't execute")
    ):
        raise GroundingError("clarification answer cancels the request")
    if _has_unsupported_condition_or_time(answer):
        raise GroundingError("clarification answer contains an unsupported condition or time")
    if _has_informational_request(answer):
        raise GroundingError("clarification answer is informational, not an authorization")
    if chosen.entity_id in grounded.negated_entity_ids:
        raise GroundingError("clarification selected an entity explicitly excluded by the user")
    if any(
        _label_is_excluded(answer_normalized, label)
        for label in (chosen.room, chosen.floor, chosen.device, *chosen.aliases)
    ):
        raise GroundingError("clarification answer explicitly excludes the selected entity")
    confirmed_value_token = _operation_value_token(confirmed.value)
    if (
        confirmed_value_token in grounded.excluded_operation_value_tokens
        or _value_is_explicitly_excluded(grounded.utterance, confirmed.value)
        or _value_is_explicitly_excluded(answer, confirmed.value)
    ):
        raise GroundingError("confirmed value was explicitly excluded by the user")
    answer_is_candidate_index = answer_normalized.isdigit()
    operational_slots = ("action", "attribute", "value", "unit")
    if answer_is_candidate_index and any(
        slot in grounded.clarification.unresolved_slots for slot in operational_slots
    ):
        raise GroundingError(
            "a candidate index cannot supply a missing or conflicting operation slot"
        )
    # A displayed index is selector UI, not an operational number.  Keeping it
    # out of value parsing prevents option 2 from becoming 2 percent/degrees.
    operational_answer = (
        "" if answer_is_candidate_index
        else _clarification_operational_text(answer, chosen)
    )
    if _has_informational_request(operational_answer):
        raise GroundingError("clarification answer is informational, not an authorization")
    if _has_unsupported_condition_or_time(operational_answer):
        raise GroundingError("clarification answer contains an unsupported condition or time")
    if _has_negative_or_cancelled_intent(operational_answer):
        raise GroundingError("clarification answer cancels the request")
    if not _clarification_has_positive_authorization(answer, chosen, grounded.candidates):
        raise GroundingError("clarification answer has no positive authorization evidence")
    answer_requirements = _answer_operational_requirements(answer, chosen)
    for slot in answer_requirements:
        if getattr(confirmed, slot) == "*":
            raise GroundingError(f"confirmed plan drops explicit {slot} from the clarification answer")
    answer_directions = _directional_actions(answer_normalized)
    confirmed_action = normalize_text(confirmed.action)
    if len(answer_directions) > 1:
        raise GroundingError("clarification answer contains opposing actions")
    if answer_directions and confirmed_action not in answer_directions:
        # "Open it to 35 percent" is a set-position answer, not a full-open
        # authorization.  Other cross-action patches fail closed.
        if not (confirmed_action == "set" and "value" in answer_requirements):
            raise GroundingError("confirmed action conflicts with the clarification answer")
    if answer_requirements and any(
        _phrase_in(answer_normalized, verb) for verb in ("set", "change", "make", "use")
    ) and confirmed_action not in {"set", "adjustup", "adjustdown"}:
        raise GroundingError("clarification answer introduces an unbound set operation")
    if not EntityRegistry._device_matches(chosen, confirmed.device):
        raise GroundingError("confirmed device does not match the selected entity")
    if normalize_text(confirmed.room) != normalize_text(chosen.room):
        raise GroundingError("confirmed room does not match the selected entity")
    if normalize_text(confirmed.floor) != normalize_text(chosen.floor):
        raise GroundingError("confirmed floor does not match the selected entity")
    confirmed_candidates = registry.candidates(confirmed)
    if tuple(entity.entity_id for entity in confirmed_candidates) != (chosen.entity_id,):
        raise GroundingError("confirmed instruction does not resolve uniquely to the selected entity")
    operational_utterance = _operational_text(
        grounded.utterance,
        grounded.candidates or registry.entities,
    )
    answer_conflicts = _operational_conflicts(operational_answer, confirmed)
    if answer_conflicts:
        detail = ", ".join(sorted(answer_conflicts))
        raise GroundingError(f"clarification answer has unresolved operational conflicts: {detail}")
    evidence = f"{operational_utterance}\n{operational_answer}"
    original_conflicts = set(_operational_conflicts(operational_utterance, confirmed))
    if not _action_supported(evidence, confirmed) and not (
        "action" in original_conflicts and _action_supported(operational_answer, confirmed)
    ):
        raise GroundingError("confirmed action is not supported by the user text")
    if not _attribute_supported_for_entity(evidence, confirmed, chosen):
        raise GroundingError("confirmed attribute is not supported by the user text")
    if not _value_supported(evidence, confirmed.value) and not (
        "value" in original_conflicts and _value_supported(operational_answer, confirmed.value)
    ):
        raise GroundingError("confirmed value is not supported by the user text")
    if not _unit_supported(evidence, confirmed) and not (
        "unit" in original_conflicts and _unit_supported(operational_answer, confirmed)
    ):
        raise GroundingError("confirmed unit is not supported by the user text")
    answer_replaces_conflicted_action = (
        "action" in original_conflicts
        and _action_supported(operational_answer, confirmed)
        and normalize_text(confirmed.action) in {"turnon", "turnoff"}
    )
    for slot in _explicit_operational_requirements(operational_utterance):
        if getattr(confirmed, slot) == "*":
            if answer_replaces_conflicted_action and slot in {"attribute", "value", "unit"}:
                continue
            raise GroundingError(f"confirmed plan drops explicit {slot} from the user request")

    def answer_supports(slot: str) -> bool:
        if answer_is_candidate_index and slot in operational_slots:
            return False
        value = getattr(confirmed, slot)
        if value == "*":
            return True
        normalized_value = normalize_text(value)
        if re.search(
            rf"\b(?:not|instead\s+of)\b(?:\s+[a-z0-9]+){{0,3}}\s+{re.escape(normalized_value)}\b",
            answer_normalized,
        ):
            return False
        if slot == "action":
            if re.search(r"\bdo\s+not\s+(?:turn|switch|open|close|set|change|make)\b", answer_normalized):
                return False
            return _action_supported(operational_answer, confirmed)
        if slot == "attribute":
            return _attribute_supported_for_entity(operational_answer, confirmed, chosen)
        if slot == "value":
            return _value_supported(operational_answer, confirmed.value)
        return _unit_supported(operational_answer, confirmed)

    # A conflicted value/action must be selected in the answer itself.  It may
    # not be assembled from unrelated clauses in the original request.
    original_normalized = normalize_text(operational_utterance)
    numbers = set(_numbers_in(original_normalized))
    if _phrase_in(original_normalized, "halfway"):
        numbers.add(50.0)
    matched_named_values = {
        value for value in (*COLOR_RGB, "cool", "heat", "dry", "fan", "auto", "low", "medium", "high")
        if _phrase_in(original_normalized, value)
    }
    named_values = {
        value for value in matched_named_values
        if not any(
            value != longer and _phrase_in(normalize_text(longer), value)
            for longer in matched_named_values
        )
    }
    conflicted_slots: set[str] = set(original_conflicts)
    if len(numbers) > 1 or len(named_values) > 1:
        conflicted_slots.add("value")
    has_on = any(_phrase_in(original_normalized, term) for term in ("turn on", "switch on", "open"))
    has_off = any(_phrase_in(original_normalized, term) for term in ("turn off", "switch off", "close"))
    if has_on and has_off:
        conflicted_slots.add("action")
    for slot in conflicted_slots:
        if not answer_supports(slot):
            raise GroundingError(f"conflicting {slot} is not independently confirmed by the answer")

    missing_slots = {
        slot
        for source in grounded.source_instructions
        for slot in _missing_required_slots(source)
    }
    for slot in missing_slots:
        if slot in operational_slots and not answer_supports(slot):
            raise GroundingError(f"missing {slot} is not supplied by the answer")

    # The confirmed operational tuple must be one source tuple plus only
    # answer-supported patches.  This prevents cross-clause Frankenstein plans.
    context = SessionContext(grounded.context_entity_ids)
    patchable = False
    for source in grounded.source_instructions:
        changed = [
            slot for slot in operational_slots
            if normalize_text(getattr(source, slot)) != normalize_text(getattr(confirmed, slot))
        ]
        if not all(answer_supports(slot) for slot in changed):
            continue
        source_candidates = registry.candidates(_source_selector(grounded.utterance, source), context)
        source_targets_chosen = any(entity.entity_id == chosen.entity_id for entity in source_candidates)
        if source_targets_chosen or all(answer_supports(slot) for slot in conflicted_slots):
            patchable = True
            break
    if not patchable:
        raise GroundingError("confirmed plan is not a valid answer-supported patch of one source instruction")


def _answer_cancels(answer_normalized: str) -> bool:
    return _has_negative_action_authorization(answer_normalized) or bool(
        re.match(r"^(?:actually\s+)?no(?:\s+thanks?)?(?:\b|[,.!])", answer_normalized)
    ) or bool(
        re.search(r"\b(?:do\s+not|don't|dont)\s*[.!]?\s*$", answer_normalized)
    ) or bool(
        re.search(
            r"\b(?:cancel(?:\s+(?:it|that|this|the\s+request))?|never\s+mind|do\s+nothing|"
            r"stop(?:\s+(?:it|that|this))?|not\s+now|wait|hold(?:\s+on)?|"
            r"(?:do\s+not|don't)\s+(?:do|proceed|act|execute)\b(?:.{0,20}\byet\b)?|"
            r"(?:do\s+not|don't|rather\s+not)\s+(?:go\s+ahead|proceed|do|act|execute|"
            r"turn|switch|open|close|set|change|make|adjust|touch)\b|"
            r"(?:i\s+)?(?:do\s+not|don't)\s+want\b|"
            r"forget\s+(?:it|that|this)|(?:i\s+)?changed?\s+my\s+mind|"
            r"not\s+(?:anymore|any\s+longer)|skip\s+(?:it|that|this)|"
            r"refrain\s+from|"
            r"(?:leave|keep)\s+(?:(?:the\s+)?[a-z0-9_.-]+\s+|it\s+|that\s+|this\s+)?"
            r"(?:on|off|open|closed|unchanged|as\s+is))\b",
            answer_normalized,
        )
    )


def _answer_is_noncommittal(answer_normalized: str) -> bool:
    return bool(re.search(
        r"\b(?:do\s+not\s+know|don't\s+know|not\s+sure|still\s+not\s+sure|unsure|"
        r"whatever|maybe|perhaps|ask\s+me\s+later|later)\b",
        answer_normalized,
    ))


def resolve_clarification_submission(
    grounded: GroundedRequest,
    *,
    answer: str,
    confirmed_instruction: DomuxInstruction,
    registry: EntityRegistry,
) -> ResolvedRequest:
    if not grounded.clarification.required:
        raise GroundingError("request is already unique; use resolve_unique_request")
    if "negative_or_cancelled_intent" in grounded.clarification.reasons:
        raise GroundingError("cancelled or negated requests cannot be confirmed from this turn")
    if "informational_request" in grounded.clarification.reasons:
        raise GroundingError("informational questions cannot authorize execution from this turn")
    if "unsupported_condition_or_time" in grounded.clarification.reasons:
        raise GroundingError("conditional or timed requests require a new immediate command")
    if "unsupported_request_grammar" in grounded.clarification.reasons:
        raise GroundingError("unsupported request language requires a new immediate command")
    if not answer.strip():
        raise GroundingError("clarification answer is empty")
    if _answer_is_noncommittal(normalize_text(answer)):
        raise GroundingError("clarification answer is noncommittal")
    if len(grounded.candidates) > len(grounded.clarification.candidates):
        raise GroundingError("candidate set is not narrow enough to present safely")
    chosen = resolve_clarification(answer, grounded.candidates)
    _validate_confirmed_instruction(grounded, answer, confirmed_instruction, chosen, registry)
    clarification_digest = digest_json({
        "request_digest": grounded.request_digest,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "chosen_entity_id": chosen.entity_id,
        "confirmed_instruction": confirmed_instruction.to_pipe(),
    })
    return ResolvedRequest(grounded, chosen, confirmed_instruction, clarification_digest)


def resolve_unique_request(grounded: GroundedRequest, registry: EntityRegistry) -> ResolvedRequest:
    if grounded.clarification.required:
        raise GroundingError("request still requires clarification")
    if len(grounded.source_instructions) != 1 or len(grounded.candidates) != 1:
        raise GroundingError("unique request invariant failed")
    chosen = grounded.candidates[0]
    confirmed = grounded.source_instructions[0]
    # A unique source may retain '*' room/floor; candidate uniqueness, not a
    # client-provided selector, binds the target in this path.
    if not EntityRegistry._device_matches(chosen, confirmed.device):
        raise GroundingError("unique request device does not match its candidate")
    clarification_digest = digest_json({
        "request_digest": grounded.request_digest,
        "answer": "unique_without_clarification",
        "chosen_entity_id": chosen.entity_id,
        "confirmed_instruction": confirmed.to_pipe(),
    })
    return ResolvedRequest(grounded, registry.get(chosen.entity_id), confirmed, clarification_digest)


@dataclass(frozen=True, init=False)
class CanonicalPlan:
    _source_slots_json: str
    entity_id: str
    domain: str
    service: str
    _service_data_json: str
    _expected_projection_json: str

    def __init__(
        self,
        *,
        source_slots: Mapping[str, str],
        entity_id: str,
        domain: str,
        service: str,
        service_data: Mapping[str, object],
        expected_projection: Mapping[str, object],
    ) -> None:
        object.__setattr__(self, "_source_slots_json", canonical_json(dict(source_slots)))
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "service", service)
        object.__setattr__(self, "_service_data_json", canonical_json(dict(service_data)))
        object.__setattr__(self, "_expected_projection_json", canonical_json(dict(expected_projection)))

    @property
    def source_slots(self) -> Mapping[str, str]:
        return json.loads(self._source_slots_json)

    @property
    def service_data(self) -> Mapping[str, object]:
        return json.loads(self._service_data_json)

    @property
    def expected_projection(self) -> Mapping[str, object]:
        return json.loads(self._expected_projection_json)

    def stable_dict(self) -> dict[str, object]:
        return {
            "source_slots": dict(self.source_slots),
            "entity_id": self.entity_id,
            "domain": self.domain,
            "service": self.service,
            "service_data": dict(self.service_data),
            "expected_projection": dict(self.expected_projection),
        }

    @property
    def digest(self) -> str:
        return digest_json(self.stable_dict())


def _numeric(value: str, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise GroundingError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise GroundingError(f"numeric value {parsed} is outside [{minimum}, {maximum}]")
    return parsed


def _require_unit(instruction: DomuxInstruction, expected: str) -> None:
    if normalize_text(instruction.unit) != normalize_text(expected):
        raise GroundingError(
            f"{instruction.attribute} requires unit {expected!r}, got {instruction.unit!r}"
        )


def _require_placeholders(instruction: DomuxInstruction, *slots: str) -> None:
    invalid = [slot for slot in slots if getattr(instruction, slot) != "*"]
    if invalid:
        raise GroundingError(
            f"{instruction.action} requires '*' for unused slots: {', '.join(invalid)}"
        )


def _require_adjust_unit(instruction: DomuxInstruction, numeric_unit: str) -> None:
    expected = "*" if instruction.value == "*" else numeric_unit
    _require_unit(instruction, expected)


def _require_temperature_alignment(
    attributes: Mapping[str, object],
    temperature: float,
    minimum: float,
) -> None:
    if "target_temp_step" not in attributes:
        return
    step = float(attributes["target_temp_step"])
    if not math.isfinite(step) or step <= 0:
        raise GroundingError("climate target_temp_step must be a positive number")
    offset_steps = (temperature - minimum) / step
    if not math.isclose(offset_steps, round(offset_steps), rel_tol=0, abs_tol=1e-8):
        raise GroundingError(
            f"temperature {temperature} does not align with the advertised {step} degree step"
        )


def controlled_projection(raw_state: Mapping[str, object], domain: str) -> dict[str, object]:
    attributes = raw_state.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    projected: dict[str, object] = {
        "entity_id": raw_state.get("entity_id"),
        "state": raw_state.get("state"),
    }
    keys = {
        "light": ("brightness", "color_temp_kelvin", "rgb_color"),
        "cover": ("current_position",),
        "climate": ("temperature", "fan_mode"),
    }[domain]
    for key in keys:
        if key in attrs:
            value = attrs[key]
            # Home Assistant's LightEntity state surface reports active color
            # attributes as null while off, even when an integration retains
            # the last value internally for the next turn-on.
            if domain == "light" and raw_state.get("state") == "off":
                value = None
            projected[key] = list(value) if isinstance(value, tuple) else value
    return projected


def planning_projection(raw_state: Mapping[str, object], domain: str) -> dict[str, object]:
    """Bind every state field that can affect ``build_plan``.

    The smaller controlled projection is used for outcome assertions.  This
    projection is deliberately separate and includes advertised capabilities,
    units, and ranges so a previously approved plan cannot outlive a capability
    change that leaves the visible value untouched.
    """

    projected = controlled_projection(raw_state, domain)
    attributes = raw_state.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    keys = {
        "light": ("supported_color_modes", "min_color_temp_kelvin", "max_color_temp_kelvin"),
        "cover": ("supported_features",),
        "climate": (
            "hvac_modes", "fan_modes", "supported_features", "temperature_unit", "min_temp", "max_temp",
            "target_temp_step",
        ),
    }[domain]
    for key in keys:
        if key in attrs:
            value = attrs[key]
            projected[key] = list(value) if isinstance(value, tuple) else value
    return projected


def validate_state_shape(raw_state: Mapping[str, object], expected_entity_id: str | None = None) -> None:
    entity_id = raw_state.get("entity_id")
    if not isinstance(entity_id, str) or not ENTITY_ID_RE.fullmatch(entity_id):
        raise AdapterError("Home Assistant state has an invalid entity_id")
    if expected_entity_id is not None and entity_id != expected_entity_id:
        raise AdapterError("Home Assistant returned state for a different entity")
    if not isinstance(raw_state.get("state"), str):
        raise AdapterError("Home Assistant state value must be a string")
    if not isinstance(raw_state.get("attributes"), Mapping):
        raise AdapterError("Home Assistant state attributes must be an object")


def build_plan(
    instruction: DomuxInstruction,
    entity: EntitySpec,
    current_state: Mapping[str, object],
) -> CanonicalPlan:
    action = normalize_text(instruction.action)
    attribute = normalize_text(instruction.attribute)
    value = instruction.value.strip()
    service: str
    service_data: dict[str, object] = {"entity_id": entity.entity_id}
    expected = controlled_projection(current_state, entity.domain)

    if entity.domain == "light":
        light_attributes = current_state.get("attributes", {})
        if not isinstance(light_attributes, Mapping):
            raise GroundingError("light attributes must be an object")
        color_modes = {
            normalize_text(mode) for mode in light_attributes.get("supported_color_modes", ())
            if isinstance(mode, str)
        }
        if action == "turnon":
            _require_placeholders(instruction, "attribute", "value", "unit")
            service, expected["state"] = "turn_on", "on"
            # HA does not expose the retained brightness/color values while a
            # light is off, so a bare turn-on can only promise the on state.
            for key in ("brightness", "color_temp_kelvin", "rgb_color"):
                expected.pop(key, None)
        elif action == "turnoff":
            _require_placeholders(instruction, "attribute", "value", "unit")
            service, expected["state"] = "turn_off", "off"
            for key in ("brightness", "color_temp_kelvin", "rgb_color"):
                if key in expected:
                    expected[key] = None
        elif action == "set" and attribute == "brightness":
            _require_unit(instruction, "Percent")
            if not color_modes.intersection({"brightness", "white", "color temp", "color_temp", "rgb", "rgbw", "rgbww", "hs", "xy"}):
                raise GroundingError("light entity does not advertise brightness support")
            percent = _numeric(value, minimum=0, maximum=100)
            service_data["brightness_pct"] = percent
            service = "turn_on"
            expected["state"] = "off" if percent == 0 else "on"
            expected["brightness"] = None if percent == 0 else round(percent * 255 / 100)
        elif action == "set" and attribute == "color":
            _require_unit(instruction, "*")
            if not color_modes.intersection({"rgb", "rgbw", "rgbww", "hs", "xy"}):
                raise GroundingError("light entity does not advertise color support")
            color = normalize_text(value)
            if color not in COLOR_RGB:
                raise GroundingError(f"unsupported light color: {value!r}")
            service_data["rgb_color"] = COLOR_RGB[color]
            service, expected["state"] = "turn_on", "on"
            expected["rgb_color"] = COLOR_RGB[color]
        elif action == "set" and attribute == "colortemperature":
            _require_unit(instruction, "Kelvin")
            if not color_modes.intersection({"color temp", "color_temp"}):
                raise GroundingError("light entity does not advertise color-temperature support")
            minimum = float(light_attributes.get("min_color_temp_kelvin", 3000))
            maximum = float(light_attributes.get("max_color_temp_kelvin", 6500))
            kelvin = _numeric(value, minimum=minimum, maximum=maximum)
            if not kelvin.is_integer():
                raise GroundingError("light color temperature must be an integer Kelvin value")
            kelvin = int(kelvin)
            service_data["color_temp_kelvin"] = kelvin
            service, expected["state"] = "turn_on", "on"
            expected["color_temp_kelvin"] = round(kelvin)
        elif action in {"adjustup", "adjustdown"} and attribute == "brightness":
            _require_adjust_unit(instruction, "Percent")
            if not color_modes.intersection({"brightness", "white", "color temp", "color_temp", "rgb", "rgbw", "rgbww", "hs", "xy"}):
                raise GroundingError("light entity does not advertise brightness support")
            before = current_state.get("attributes", {})
            current = float(before.get("brightness", 0)) * 100 / 255
            step = 10.0 if value == "*" else _numeric(value, minimum=0, maximum=100)
            target = min(100.0, current + step) if action == "adjustup" else max(0.0, current - step)
            service_data["brightness_pct"] = round(target, 2)
            service = "turn_on"
            expected["state"] = "off" if target == 0 else "on"
            expected["brightness"] = None if target == 0 else round(target * 255 / 100)
        else:
            raise GroundingError(f"unsupported light operation: {instruction.to_pipe()}")
    elif entity.domain == "cover":
        cover_attributes = current_state.get("attributes", {})
        if not isinstance(cover_attributes, Mapping):
            raise GroundingError("cover attributes must be an object")
        supported_features = int(cover_attributes.get("supported_features", 0))
        if action == "turnon":
            _require_placeholders(instruction, "attribute", "value", "unit")
            if not (supported_features & 1):
                raise GroundingError("cover entity does not advertise open support")
            service, expected["state"] = "open_cover", "open"
            if "current_position" in expected:
                expected["current_position"] = 100
        elif action == "turnoff":
            _require_placeholders(instruction, "attribute", "value", "unit")
            if not (supported_features & 2):
                raise GroundingError("cover entity does not advertise close support")
            service, expected["state"] = "close_cover", "closed"
            if "current_position" in expected:
                expected["current_position"] = 0
        elif action == "set" and attribute in {"position", "openness"}:
            _require_unit(instruction, "Percent")
            if not (supported_features & 4):
                raise GroundingError("cover entity does not advertise position support")
            position = _numeric(value, minimum=0, maximum=100)
            if not position.is_integer():
                raise GroundingError("cover position must be an integer percent")
            position = int(position)
            service_data["position"] = position
            service = "set_cover_position"
            expected["state"] = "closed" if position == 0 else "open"
            expected["current_position"] = round(position)
        elif action in {"adjustup", "adjustdown"} and attribute in {"position", "openness"}:
            _require_adjust_unit(instruction, "Percent")
            if not (supported_features & 4):
                raise GroundingError("cover entity does not advertise position support")
            if "current_position" not in cover_attributes:
                raise GroundingError("cover adjustment requires an observed current_position")
            current = float(cover_attributes["current_position"])
            if not math.isfinite(current) or not current.is_integer():
                raise GroundingError("cover current_position must be an integer percent")
            current = int(current)
            step = 10.0 if value == "*" else _numeric(value, minimum=0, maximum=100)
            if not step.is_integer():
                raise GroundingError("cover position adjustment must be an integer percent")
            step = int(step)
            target = min(100.0, current + step) if action == "adjustup" else max(0.0, current - step)
            target = int(target)
            service_data["position"] = target
            service = "set_cover_position"
            expected["state"] = "closed" if target == 0 else "open"
            expected["current_position"] = round(target)
        else:
            raise GroundingError(f"unsupported cover operation: {instruction.to_pipe()}")
    else:
        attributes = current_state.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise GroundingError("climate attributes must be an object")
        hvac_modes = {
            normalize_text(mode): mode for mode in attributes.get("hvac_modes", ())
            if isinstance(mode, str)
        }
        fan_modes = {
            normalize_text(mode): mode for mode in attributes.get("fan_modes", ())
            if isinstance(mode, str)
        }
        supported_features = int(attributes.get("supported_features", 0))
        temperature_unit = str(attributes.get("temperature_unit", ""))
        if not hvac_modes:
            raise GroundingError("climate entity does not advertise hvac_modes")
        if action == "turnon":
            _require_placeholders(instruction, "attribute", "value", "unit")
            active_modes = [value for key, value in hvac_modes.items() if key != "off"]
            if len(active_modes) != 1:
                raise GroundingError(
                    "climate turnOn requires exactly one advertised active mode; confirm a mode explicitly"
                )
            service_data["hvac_mode"] = active_modes[0]
            service, expected["state"] = "set_hvac_mode", active_modes[0]
        elif action == "turnoff":
            _require_placeholders(instruction, "attribute", "value", "unit")
            if "off" in hvac_modes:
                service = "set_hvac_mode"
                service_data["hvac_mode"] = hvac_modes["off"]
            elif supported_features & 128:
                # Some integrations expose TURN_OFF as a feature instead of
                # listing off in hvac_modes.
                service = "turn_off"
            else:
                raise GroundingError("climate entity does not advertise turn-off support")
            expected["state"] = "off"
        elif action == "set" and attribute == "temperature":
            _require_unit(instruction, "Celsius")
            if temperature_unit not in {"°C", "C", "Celsius"}:
                raise GroundingError("only Celsius climate entities are supported")
            if not (supported_features & 1):
                raise GroundingError("climate entity does not support target temperature")
            minimum = float(attributes.get("min_temp", 16))
            maximum = float(attributes.get("max_temp", 30))
            temperature = _numeric(value, minimum=minimum, maximum=maximum)
            _require_temperature_alignment(attributes, temperature, minimum)
            service_data["temperature"] = temperature
            service = "set_temperature"
            expected["temperature"] = temperature
        elif action == "set" and attribute == "mode":
            _require_unit(instruction, "*")
            mode = normalize_text(value)
            mode = "fan only" if mode == "fan" else mode
            if mode not in hvac_modes or mode == "off":
                raise GroundingError(f"unsupported climate mode: {value!r}")
            advertised_mode = hvac_modes[mode]
            service_data["hvac_mode"] = advertised_mode
            service = "set_hvac_mode"
            expected["state"] = advertised_mode
        elif action == "set" and attribute in {"windspeed", "wind speed", "fan speed"}:
            _require_unit(instruction, "Level")
            fan_mode = normalize_text(value)
            if not (supported_features & 8) or fan_mode not in fan_modes:
                raise GroundingError(f"unsupported fan mode: {value!r}")
            advertised_fan_mode = fan_modes[fan_mode]
            service_data["fan_mode"] = advertised_fan_mode
            service = "set_fan_mode"
            expected["fan_mode"] = advertised_fan_mode
        elif action in {"adjustup", "adjustdown"} and attribute == "temperature":
            _require_adjust_unit(instruction, "Celsius")
            if temperature_unit not in {"°C", "C", "Celsius"} or not (supported_features & 1):
                raise GroundingError("climate entity does not support Celsius target temperature")
            current = float(current_state.get("attributes", {}).get("temperature", 24))
            if not math.isfinite(current):
                raise GroundingError("current climate temperature must be finite")
            step = 1.0 if value == "*" else _numeric(value, minimum=0, maximum=14)
            minimum = float(attributes.get("min_temp", 16))
            maximum = float(attributes.get("max_temp", 30))
            target = min(maximum, current + step) if action == "adjustup" else max(minimum, current - step)
            _require_temperature_alignment(attributes, target, minimum)
            service_data["temperature"] = target
            service = "set_temperature"
            expected["temperature"] = target
        else:
            raise GroundingError(f"unsupported climate operation: {instruction.to_pipe()}")

    return CanonicalPlan(
        source_slots=instruction.canonical_slots(),
        entity_id=entity.entity_id,
        domain=entity.domain,
        service=service,
        service_data=service_data,
        expected_projection=expected,
    )


def projection_matches(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    """Match all operation-controlled fields while allowing extra HA state."""

    if not set(expected).issubset(actual):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float) and isinstance(actual_value, (int, float)):
            if not math.isclose(float(actual_value), expected_value, rel_tol=0, abs_tol=0.01):
                return False
        elif actual_value != expected_value:
            return False
    return True


class InMemoryHAAdapter:
    """Deterministic adapter used for policy evaluation and offline replay."""

    def __init__(self, states: Mapping[str, Mapping[str, object]]):
        self._states = json.loads(json.dumps(states))
        for entity_id, state in self._states.items():
            validate_state_shape(state, entity_id)
        self.sut_calls: list[dict[str, object]] = []
        self.setup_calls: list[dict[str, object]] = []
        self.force_postcondition_mismatch = False

    def get_state(self, entity_id: str) -> dict[str, object]:
        try:
            state = json.loads(json.dumps(self._states[entity_id]))
        except KeyError as exc:
            raise AdapterError(f"state not found for allowed entity: {entity_id}") from exc
        validate_state_shape(state, entity_id)
        return state

    def set_state_for_setup(self, entity_id: str, state: Mapping[str, object]) -> None:
        validate_state_shape(state, entity_id)
        self._states[entity_id] = json.loads(json.dumps(state))
        self.setup_calls.append({"kind": "setup", "entity_id": entity_id})

    def mutate_state_for_setup(self, entity_id: str) -> None:
        state = self.get_state(entity_id)
        domain = entity_id.split(".", 1)[0]
        if domain == "light":
            state["state"] = "off" if state.get("state") == "on" else "on"
        elif domain == "cover":
            current = int(state.get("attributes", {}).get("current_position", 0))
            updated = 100 - current
            state.setdefault("attributes", {})["current_position"] = updated
            state["state"] = "closed" if updated == 0 else "open"
        else:
            current = float(state.get("attributes", {}).get("temperature", 24))
            state.setdefault("attributes", {})["temperature"] = 25 if current != 25 else 24
        self.set_state_for_setup(entity_id, state)

    def call_service(self, domain: str, service: str, data: Mapping[str, object]) -> ServiceCallResult:
        entity_id = str(data["entity_id"])
        if not entity_id.startswith(f"{domain}."):
            raise ServiceCallError(
                "service domain does not match entity",
                attempted=False,
                acknowledged=False,
                outcome_unknown=False,
            )
        try:
            before = self.get_state(entity_id)
        except Exception as exc:
            raise ServiceCallError(
                "state read failed before in-memory dispatch",
                attempted=False,
                acknowledged=False,
                outcome_unknown=False,
            ) from exc
        after = self.get_state(entity_id)
        attrs = after.setdefault("attributes", {})
        event = {
            "kind": "sut",
            "domain": domain,
            "service": service,
            "data": dict(data),
            "before": controlled_projection(before, domain),
            "after": None,
            "acknowledged": False,
            "outcome": "attempted",
        }
        self.sut_calls.append(event)
        try:
            if domain == "light":
                if service == "turn_on":
                    if "brightness_pct" in data:
                        brightness_pct = float(data["brightness_pct"])
                        if brightness_pct == 0:
                            after["state"] = "off"
                        else:
                            after["state"] = "on"
                            attrs["brightness"] = round(brightness_pct * 255 / 100)
                    else:
                        after["state"] = "on"
                    if "rgb_color" in data:
                        attrs["rgb_color"] = list(data["rgb_color"])
                    if "color_temp_kelvin" in data:
                        attrs["color_temp_kelvin"] = round(float(data["color_temp_kelvin"]))
                elif service == "turn_off":
                    after["state"] = "off"
                else:
                    raise AdapterError(f"unsupported light service: {service}")
            elif domain == "cover":
                if service == "open_cover":
                    after["state"], attrs["current_position"] = "open", 100
                elif service == "close_cover":
                    after["state"], attrs["current_position"] = "closed", 0
                elif service == "set_cover_position":
                    position = round(float(data["position"]))
                    after["state"] = "closed" if position == 0 else "open"
                    attrs["current_position"] = position
                else:
                    raise AdapterError(f"unsupported cover service: {service}")
            elif domain == "climate":
                if service == "turn_on":
                    after["state"] = "cool"
                elif service == "turn_off":
                    after["state"] = "off"
                elif service == "set_temperature":
                    attrs["temperature"] = float(data["temperature"])
                elif service == "set_hvac_mode":
                    after["state"] = data["hvac_mode"]
                elif service == "set_fan_mode":
                    attrs["fan_mode"] = data["fan_mode"]
                else:
                    raise AdapterError(f"unsupported climate service: {service}")
            else:
                raise AdapterError(f"unsupported service domain: {domain}")
        except Exception as exc:
            event["outcome"] = "rejected_before_acknowledgement"
            raise ServiceCallError(
                "in-memory service rejected the request",
                attempted=True,
                acknowledged=False,
                outcome_unknown=False,
            ) from exc
        if not self.force_postcondition_mismatch:
            self._states[entity_id] = after
        observed = self.get_state(entity_id)
        event["acknowledged"] = True
        event["outcome"] = "observed"
        event["after"] = controlled_projection(observed, domain)
        return ServiceCallResult(
            after=observed,
            attempted=True,
            acknowledged=True,
            outcome_unknown=False,
        )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class _RequestFailure(AdapterError):
    def __init__(
        self,
        message: str,
        *,
        response_received: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.response_received = response_received
        self.status_code = status_code


class HomeAssistantRESTAdapter:
    """Minimal client for Home Assistant's official REST API.

    The default loopback restriction prevents an example token from being used
    against a public or production Home Assistant instance by accident.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 10.0,
        poll_seconds: float = 5.0,
        allow_non_loopback: bool = False,
    ):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not allow_non_loopback and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("only loopback Home Assistant URLs are allowed by default")
        if allow_non_loopback and parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and parsed.scheme != "https":
            raise ValueError("non-loopback Home Assistant URLs require HTTPS")
        if not token:
            raise ValueError("Home Assistant token is required")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._poll_seconds = poll_seconds
        self._opener = urllib.request.build_opener(_NoRedirect())
        self.sut_calls: list[dict[str, object]] = []

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> object:
        body = None if payload is None else canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            exc.close()
            raise _RequestFailure(
                f"Home Assistant REST request failed for {method} {path}",
                response_received=True,
                status_code=status_code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _RequestFailure(
                f"Home Assistant REST request failed for {method} {path}",
                response_received=False,
            ) from exc
        return json.loads(raw) if raw else None

    def get_state(self, entity_id: str) -> dict[str, object]:
        encoded = urllib.parse.quote(entity_id, safe=".")
        result = self._request("GET", f"/api/states/{encoded}")
        if not isinstance(result, dict):
            raise AdapterError("Home Assistant returned a non-object state")
        validate_state_shape(result, entity_id)
        if entity_id.startswith("climate.") and "temperature_unit" not in result["attributes"]:
            config = self._request("GET", "/api/config")
            if not isinstance(config, dict):
                raise AdapterError("Home Assistant returned a non-object config")
            unit_system = config.get("unit_system")
            if not isinstance(unit_system, Mapping):
                raise AdapterError("Home Assistant config has no unit_system object")
            temperature_unit = unit_system.get("temperature")
            if not isinstance(temperature_unit, str) or not temperature_unit.strip():
                raise AdapterError("Home Assistant config has no temperature unit")
            result = copy.deepcopy(result)
            attributes = result["attributes"]
            if not isinstance(attributes, dict):
                attributes = dict(attributes)
                result["attributes"] = attributes
            attributes["temperature_unit"] = temperature_unit
        return result

    def call_service(self, domain: str, service: str, data: Mapping[str, object]) -> ServiceCallResult:
        entity_id = str(data["entity_id"])
        if not entity_id.startswith(f"{domain}."):
            raise ServiceCallError(
                "service domain does not match entity",
                attempted=False,
                acknowledged=False,
                outcome_unknown=False,
            )
        try:
            before = self.get_state(entity_id)
        except Exception as exc:
            raise ServiceCallError(
                "Home Assistant state read failed before dispatch",
                attempted=False,
                acknowledged=False,
                outcome_unknown=False,
            ) from exc
        event = {
            "kind": "sut",
            "domain": domain,
            "service": service,
            "data": dict(data),
            "before": controlled_projection(before, domain),
            "after": None,
            "acknowledged": False,
            "outcome": "attempted",
        }
        self.sut_calls.append(event)
        try:
            self._request("POST", f"/api/services/{domain}/{service}", data)
        except _RequestFailure as exc:
            # A 4xx is an explicit rejection.  A 5xx can arrive after HA (or an
            # integration it called) has already applied a side effect, so the
            # exact outcome remains unknown even though an HTTP response exists.
            unknown = not exc.response_received or (
                exc.status_code is not None and 500 <= exc.status_code < 600
            )
            event["outcome"] = "request_error_outcome_unknown" if unknown else "request_rejected"
            raise ServiceCallError(
                "Home Assistant service request failed",
                attempted=True,
                acknowledged=False,
                outcome_unknown=unknown,
            ) from exc
        event["acknowledged"] = True
        event["outcome"] = "acknowledged_state_pending"
        deadline = time.monotonic() + self._poll_seconds
        try:
            after = self.get_state(entity_id)
            while time.monotonic() < deadline and after == before:
                time.sleep(0.1)
                after = self.get_state(entity_id)
        except Exception as exc:
            event["outcome"] = "acknowledged_state_unknown"
            raise ServiceCallError(
                "Home Assistant acknowledged the service but state observation failed",
                attempted=True,
                acknowledged=True,
                outcome_unknown=True,
            ) from exc
        event["after"] = controlled_projection(after, domain)
        event["outcome"] = "observed"
        return ServiceCallResult(
            after=after,
            attempted=True,
            acknowledged=True,
            outcome_unknown=False,
        )

    def wait_for_projection(
        self,
        entity_id: str,
        domain: str,
        expected: Mapping[str, object],
    ) -> dict[str, object]:
        """Poll through transient HA states until the approved projection is visible."""

        deadline = time.monotonic() + self._poll_seconds
        latest = self.get_state(entity_id)
        while not projection_matches(controlled_projection(latest, domain), expected):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            latest = self.get_state(entity_id)
        return latest


def state_binding(
    adapter: Any,
    registry: EntityRegistry,
    entity_ids: Iterable[str],
    *,
    for_planning: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for entity_id in sorted(set(entity_ids)):
        entity = registry.get(entity_id)
        raw = adapter.get_state(entity_id)
        result[entity_id] = (
            planning_projection(raw, entity.domain)
            if for_planning
            else controlled_projection(raw, entity.domain)
        )
    return result


@dataclass(frozen=True)
class Confirmation:
    actor_id: str
    session_id: str
    nonce: str
    request_digest: str
    clarification_digest: str
    plan_digest: str
    candidate_digest: str


@dataclass(frozen=True)
class PreparedAction:
    actor_id: str
    session_id: str
    nonce: str
    request_digest: str
    clarification_digest: str
    plan_digest: str
    candidate_digest: str
    entity_id: str
    created_at: float
    expires_at: float

    def confirmation(self) -> Confirmation:
        return Confirmation(
            actor_id=self.actor_id,
            session_id=self.session_id,
            nonce=self.nonce,
            request_digest=self.request_digest,
            clarification_digest=self.clarification_digest,
            plan_digest=self.plan_digest,
            candidate_digest=self.candidate_digest,
        )


@dataclass
class _StoredAction:
    actor_id: str
    session_id: str
    nonce: str
    utterance: str
    raw_output: str
    context_entity_ids: tuple[str, ...]
    request_digest: str
    clarification_digest: str
    plan: CanonicalPlan
    plan_digest: str
    candidate_ids: tuple[str, ...]
    state_entity_ids: tuple[str, ...]
    candidate_digest: str
    state_digest: str
    created_at: float
    expires_at: float
    status: str = "PREPARED"
    consumed: bool = False


@dataclass(frozen=True)
class _ActionTombstone:
    nonce: str
    plan_digest: str
    candidate_digest: str
    created_at: float
    expires_at: float
    status: str
    consumed: bool


@dataclass(frozen=True)
class CommitResult:
    accepted: bool
    dispatched: bool
    reason: str
    status: str
    nonce: str
    plan_digest: str | None = None
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None
    acknowledged: bool = False
    outcome_unknown: bool = False
    before_registry_digest: str | None = None
    after_registry_digest: str | None = None


class PreparedActionStore:
    """Thread-safe one-time authorization store for a single process."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        nonce_factory: Callable[[], str] = lambda: secrets.token_urlsafe(24),
        max_items: int = 10_000,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._ttl = ttl_seconds
        self._clock = clock
        self._nonce_factory = nonce_factory
        self._max_items = max_items
        self._items: dict[str, _StoredAction | _ActionTombstone] = {}
        self._entity_locks: dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    def prepare(
        self,
        *,
        actor_id: str,
        session_id: str,
        grounded: GroundedRequest,
        registry: EntityRegistry,
        adapter: Any,
        clarification_answer: str | None = None,
        confirmed_instruction: DomuxInstruction | None = None,
        state_dependencies: Sequence[str] = (),
    ) -> PreparedAction:
        if not actor_id or not session_id:
            raise ValueError("actor_id and session_id are required")
        context = SessionContext(grounded.context_entity_ids)
        server_grounded = ground_domux_request(
            grounded.utterance,
            grounded.raw_output,
            registry,
            context,
        )
        if server_grounded.request_digest != grounded.request_digest:
            raise GroundingError("grounded request does not match the server-side reconstruction")
        if server_grounded.clarification.required:
            if clarification_answer is None or confirmed_instruction is None:
                raise GroundingError("clarification answer and complete confirmed instruction are required")
            resolved = resolve_clarification_submission(
                server_grounded,
                answer=clarification_answer,
                confirmed_instruction=confirmed_instruction,
                registry=registry,
            )
        else:
            if clarification_answer is not None or confirmed_instruction is not None:
                raise GroundingError("a unique request cannot be replaced by client confirmation fields")
            resolved = resolve_unique_request(server_grounded, registry)

        candidate_ids = tuple(sorted(entity.entity_id for entity in server_grounded.candidates))
        chosen = registry.get(resolved.chosen.entity_id)
        context_entity_ids = tuple(server_grounded.context_entity_ids)
        state_entity_ids = tuple(sorted({chosen.entity_id, *context_entity_ids, *state_dependencies}))
        for entity_id in state_entity_ids:
            registry.get(entity_id)
        raw_states: dict[str, Mapping[str, object]] = {}
        for entity_id in state_entity_ids:
            raw_states[entity_id] = adapter.get_state(entity_id)
        plan = build_plan(resolved.confirmed_instruction, chosen, raw_states[chosen.entity_id])
        plan_digest = plan.digest
        candidate_digest = registry.metadata_digest(candidate_ids)
        bound_projection = {
            entity_id: planning_projection(raw_states[entity_id], registry.get(entity_id).domain)
            for entity_id in state_entity_ids
        }
        state_digest = digest_json(bound_projection)
        created = self._clock()
        nonce = self._nonce_factory()
        if not nonce:
            raise ValueError("nonce factory returned an empty value")
        action = _StoredAction(
            actor_id=actor_id,
            session_id=session_id,
            nonce=nonce,
            utterance=server_grounded.utterance,
            raw_output=server_grounded.raw_output,
            context_entity_ids=context_entity_ids,
            request_digest=server_grounded.request_digest,
            clarification_digest=resolved.clarification_digest,
            plan=plan,
            plan_digest=plan_digest,
            candidate_ids=candidate_ids,
            state_entity_ids=state_entity_ids,
            candidate_digest=candidate_digest,
            state_digest=state_digest,
            created_at=created,
            expires_at=created + self._ttl,
        )
        with self._lock:
            self._purge_expired_locked(created)
            if len(self._items) >= self._max_items:
                now = self._clock()
                removable = [
                    key for key, item in self._items.items()
                    if item.consumed or item.status != "PREPARED" or now > item.expires_at
                ]
                for key in removable:
                    self._items.pop(key, None)
            if len(self._items) >= self._max_items:
                raise RuntimeError("prepared-action store capacity exceeded")
            if nonce in self._items:
                raise ValueError("nonce collision")
            self._items[nonce] = action
        return PreparedAction(
            actor_id=actor_id,
            session_id=session_id,
            nonce=nonce,
            request_digest=action.request_digest,
            clarification_digest=action.clarification_digest,
            plan_digest=action.plan_digest,
            candidate_digest=action.candidate_digest,
            entity_id=action.plan.entity_id,
            created_at=action.created_at,
            expires_at=action.expires_at,
        )

    def _reject(
        self,
        action: _StoredAction | None,
        nonce: str,
        reason: str,
        status: str,
        *,
        mutate: bool = True,
    ) -> CommitResult:
        if action is not None and mutate:
            action.status = status
        result = CommitResult(
            accepted=False,
            dispatched=False,
            reason=reason,
            status=status,
            nonce=nonce,
            plan_digest=None if action is None else action.plan_digest,
        )
        if action is not None and mutate and status != "PREPARED":
            with self._lock:
                self._redact_locked(action)
        return result

    def _redact_locked(self, action: _StoredAction) -> None:
        """Replace terminal actions with a digest-only replay tombstone."""

        if self._items.get(action.nonce) is not action:
            return
        self._items[action.nonce] = _ActionTombstone(
            nonce=action.nonce,
            plan_digest=action.plan_digest,
            candidate_digest=action.candidate_digest,
            created_at=action.created_at,
            expires_at=action.expires_at,
            status=action.status,
            consumed=action.consumed,
        )

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            item for item in self._items.values()
            if isinstance(item, _StoredAction)
            and item.status == "PREPARED"
            and now > item.expires_at
        ]
        for item in expired:
            item.status = "EXPIRED"
            self._redact_locked(item)
        return len(expired)

    def purge_expired(self) -> int:
        """Redact abandoned prepared requests; applications may call this from a timer."""

        with self._lock:
            return self._purge_expired_locked(self._clock())

    @staticmethod
    def _tombstone_result(action: _ActionTombstone) -> CommitResult:
        return CommitResult(
            accepted=False,
            dispatched=False,
            reason="replayed_nonce" if action.consumed else "action_not_prepared",
            status=action.status,
            nonce=action.nonce,
            plan_digest=action.plan_digest,
        )

    def _confirmation_error(
        self,
        action: _StoredAction,
        confirmation: Confirmation,
        *,
        enforce_lifecycle: bool,
    ) -> str | None:
        if enforce_lifecycle and action.consumed:
            return "replayed_nonce"
        if enforce_lifecycle and action.status != "PREPARED":
            return "action_not_prepared"
        checks = (
            (confirmation.actor_id, action.actor_id, "actor_mismatch"),
            (confirmation.session_id, action.session_id, "session_mismatch"),
            (confirmation.request_digest, action.request_digest, "request_mismatch"),
            (confirmation.clarification_digest, action.clarification_digest, "clarification_mismatch"),
            (confirmation.plan_digest, action.plan_digest, "plan_mismatch"),
            (confirmation.candidate_digest, action.candidate_digest, "confirmation_candidate_mismatch"),
        )
        return next((reason for actual, expected, reason in checks if actual != expected), None)

    def _execute(
        self,
        action: _StoredAction,
        *,
        registry: EntityRegistry,
        adapter: Any,
        before_all: Mapping[str, Mapping[str, object]],
    ) -> CommitResult:
        before = dict(before_all[action.plan.entity_id])
        before_digest = digest_json(before_all)
        try:
            receipt = adapter.call_service(
                action.plan.domain,
                action.plan.service,
                action.plan.service_data,
            )
            if not isinstance(receipt, ServiceCallResult):
                raise ServiceCallError(
                    "adapter violated the ServiceCallResult contract",
                    attempted=True,
                    acknowledged=False,
                    outcome_unknown=True,
                )
            after_raw = receipt.after
        except ServiceCallError as exc:
            action.status = "FAILED_DISPATCH"
            return CommitResult(
                accepted=True,
                dispatched=exc.attempted,
                acknowledged=exc.acknowledged,
                outcome_unknown=exc.outcome_unknown,
                reason="dispatch_failed",
                status=action.status,
                nonce=action.nonce,
                plan_digest=action.plan_digest,
                before=before,
                after=None,
                before_registry_digest=before_digest,
            )
        except Exception:
            action.status = "FAILED_DISPATCH"
            return CommitResult(
                accepted=True,
                dispatched=True,
                acknowledged=False,
                outcome_unknown=True,
                reason="dispatch_failed",
                status=action.status,
                nonce=action.nonce,
                plan_digest=action.plan_digest,
                before=before,
                after=None,
                before_registry_digest=before_digest,
            )
        wait_for_projection = getattr(adapter, "wait_for_projection", None)
        if callable(wait_for_projection):
            try:
                after_raw = wait_for_projection(
                    action.plan.entity_id,
                    action.plan.domain,
                    action.plan.expected_projection,
                )
            except Exception:
                action.status = "FAILED_POSTCONDITION"
                return CommitResult(
                    accepted=True,
                    dispatched=receipt.attempted,
                    acknowledged=receipt.acknowledged,
                    outcome_unknown=True,
                    reason="postcondition_state_unknown",
                    status=action.status,
                    nonce=action.nonce,
                    plan_digest=action.plan_digest,
                    before=before,
                    after=controlled_projection(after_raw, action.plan.domain),
                    before_registry_digest=before_digest,
                )
        try:
            after_all = state_binding(adapter, registry, (entity.entity_id for entity in registry.entities))
        except Exception:
            action.status = "FAILED_POSTCONDITION"
            return CommitResult(
                accepted=True,
                dispatched=receipt.attempted,
                acknowledged=receipt.acknowledged,
                outcome_unknown=True,
                reason="postcondition_state_unknown",
                status=action.status,
                nonce=action.nonce,
                plan_digest=action.plan_digest,
                before=before,
                after=controlled_projection(after_raw, action.plan.domain),
                before_registry_digest=before_digest,
            )
        after = after_all[action.plan.entity_id]
        exact = set(after_all) == set(before_all) and all(
            (
                projection_matches(after_all[entity_id], action.plan.expected_projection)
                if entity_id == action.plan.entity_id
                else after_all[entity_id] == before_all[entity_id]
            )
            for entity_id in before_all
        )
        after_digest = digest_json(after_all)
        if not exact:
            action.status = "FAILED_POSTCONDITION"
            return CommitResult(
                accepted=True,
                dispatched=receipt.attempted,
                acknowledged=receipt.acknowledged,
                outcome_unknown=receipt.outcome_unknown,
                reason="postcondition_mismatch",
                status=action.status,
                nonce=action.nonce,
                plan_digest=action.plan_digest,
                before=before,
                after=after,
                before_registry_digest=before_digest,
                after_registry_digest=after_digest,
            )
        action.status = "COMMITTED"
        return CommitResult(
            accepted=True,
            dispatched=receipt.attempted,
            acknowledged=receipt.acknowledged,
            outcome_unknown=receipt.outcome_unknown,
            reason="committed",
            status=action.status,
            nonce=action.nonce,
            plan_digest=action.plan_digest,
            before=before,
            after=after,
            before_registry_digest=before_digest,
            after_registry_digest=after_digest,
        )

    def commit(
        self,
        confirmation: Confirmation,
        *,
        registry: EntityRegistry,
        adapter: Any,
    ) -> CommitResult:
        with self._lock:
            action = self._items.get(confirmation.nonce)
            if action is None:
                return self._reject(None, confirmation.nonce, "unknown_nonce", "INVALIDATED")
            if isinstance(action, _ActionTombstone):
                return self._tombstone_result(action)
            error = self._confirmation_error(action, confirmation, enforce_lifecycle=True)
            if error:
                return self._reject(action, action.nonce, error, action.status, mutate=False)
            if self._clock() > action.expires_at:
                return self._reject(action, action.nonce, "expired", "EXPIRED")
            # The postcondition asserts that no registered entity changed as a
            # side effect.  Lock that same in-process scope, in sorted order,
            # so concurrent commits cannot invalidate one another's evidence.
            lock_entity_ids = tuple(sorted(entity.entity_id for entity in registry.entities))
            entity_locks = tuple(
                self._entity_locks.setdefault(entity_id, threading.Lock())
                for entity_id in lock_entity_ids
            )

        with ExitStack() as lock_stack:
            for entity_lock in entity_locks:
                lock_stack.enter_context(entity_lock)
            with self._lock:
                action = self._items.get(confirmation.nonce)
                if action is None:
                    return self._reject(None, confirmation.nonce, "unknown_nonce", "INVALIDATED")
                if isinstance(action, _ActionTombstone):
                    return self._tombstone_result(action)
                error = self._confirmation_error(action, confirmation, enforce_lifecycle=True)
                if error:
                    return self._reject(action, action.nonce, error, action.status, mutate=False)
                if self._clock() > action.expires_at:
                    return self._reject(action, action.nonce, "expired", "EXPIRED")

            context = SessionContext(action.context_entity_ids)
            regrounded = ground_domux_request(action.utterance, action.raw_output, registry, context)
            current_candidate_ids = tuple(sorted(entity.entity_id for entity in regrounded.candidates))
            current_candidate_digest = registry.metadata_digest(current_candidate_ids)
            if (
                current_candidate_ids != action.candidate_ids
                or current_candidate_digest != action.candidate_digest
            ):
                with self._lock:
                    return self._reject(action, action.nonce, "candidate_set_changed", "INVALIDATED")
            try:
                before_all = state_binding(
                    adapter,
                    registry,
                    (entity.entity_id for entity in registry.entities),
                )
            except Exception:
                return CommitResult(
                    accepted=False,
                    dispatched=False,
                    reason="predispatch_state_read_failed",
                    status=action.status,
                    nonce=action.nonce,
                    plan_digest=action.plan_digest,
                )
            try:
                current_bound = state_binding(
                    adapter,
                    registry,
                    action.state_entity_ids,
                    for_planning=True,
                )
            except Exception:
                return CommitResult(
                    accepted=False,
                    dispatched=False,
                    reason="predispatch_state_read_failed",
                    status=action.status,
                    nonce=action.nonce,
                    plan_digest=action.plan_digest,
                )
            current_state_digest = digest_json(current_bound)
            if current_state_digest != action.state_digest:
                with self._lock:
                    return self._reject(action, action.nonce, "state_changed", "INVALIDATED")
            with self._lock:
                error = self._confirmation_error(action, confirmation, enforce_lifecycle=True)
                if error:
                    return self._reject(action, action.nonce, error, action.status, mutate=False)
                if self._clock() > action.expires_at:
                    return self._reject(action, action.nonce, "expired", "EXPIRED")
                action.consumed = True
                action.status = "DISPATCHING"
            result = self._execute(action, registry=registry, adapter=adapter, before_all=before_all)
            with self._lock:
                self._redact_locked(action)
            return result

    def snapshot(self, nonce: str) -> dict[str, object]:
        with self._lock:
            action = self._items[nonce]
            if isinstance(action, _ActionTombstone):
                return {
                    "nonce": action.nonce,
                    "plan_digest": action.plan_digest,
                    "candidate_digest": action.candidate_digest,
                    "created_at": action.created_at,
                    "expires_at": action.expires_at,
                    "status": action.status,
                    "consumed": action.consumed,
                    "redacted": True,
                }
            return {
                "actor_id": action.actor_id,
                "session_id": action.session_id,
                "nonce": action.nonce,
                "request_digest": action.request_digest,
                "clarification_digest": action.clarification_digest,
                "plan_digest": action.plan_digest,
                "candidate_ids": list(action.candidate_ids),
                "state_entity_ids": list(action.state_entity_ids),
                "candidate_digest": action.candidate_digest,
                "state_digest": action.state_digest,
                "created_at": action.created_at,
                "expires_at": action.expires_at,
                "status": action.status,
                "consumed": action.consumed,
                "plan": action.plan.stable_dict(),
                "redacted": False,
            }


class ClarifyPrepareStore(PreparedActionStore):
    """Credible baseline: server plan/session binding without temporal guards.

    It deliberately does not revalidate candidate metadata, relevant state, TTL,
    or one-time use.  Those are the only independent variables added by the
    Clarify-and-Commit store above.
    """

    def commit(
        self,
        confirmation: Confirmation,
        *,
        registry: EntityRegistry,
        adapter: Any,
    ) -> CommitResult:
        with self._lock:
            action = self._items.get(confirmation.nonce)
            if action is None:
                return self._reject(None, confirmation.nonce, "unknown_nonce", "INVALIDATED")
            error = self._confirmation_error(action, confirmation, enforce_lifecycle=False)
            if error:
                return self._reject(action, action.nonce, error, action.status, mutate=False)
            action.status = "DISPATCHING"
        try:
            before_all = state_binding(
                adapter,
                registry,
                (entity.entity_id for entity in registry.entities),
            )
        except Exception:
            return CommitResult(
                accepted=False,
                dispatched=False,
                reason="predispatch_state_read_failed",
                status=action.status,
                nonce=action.nonce,
                plan_digest=action.plan_digest,
            )
        return self._execute(action, registry=registry, adapter=adapter, before_all=before_all)


def altered_confirmation(confirmation: Confirmation, **changes: object) -> Confirmation:
    """Small explicit helper used by mutation tests and the frozen evaluator."""

    return replace(confirmation, **changes)
