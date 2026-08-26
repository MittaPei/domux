#!/usr/bin/env python3
"""Run a reproducible, isolated Home Assistant REST acceptance check.

The runner creates exactly one task-labelled container and one task-labelled
named volume. Synthetic credentials and Home Assistant tokens remain in process
memory and are deliberately excluded from the deterministic JSON artifact.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from clarify_commit import (
    DomuxInstruction,
    EntityRegistry,
    EntitySpec,
    HomeAssistantRESTAdapter,
    PreparedActionStore,
    ground_domux_request,
    parse_domux_output,
    projection_matches,
    resolve_clarification_submission,
    resolve_unique_request,
)


IMAGE_REPOSITORY = "ghcr.io/home-assistant/home-assistant"
IMAGE_DIGEST = "sha256:8e9751cb66d3ba6624f5360a7d31b0c6821f7f5b3fb8ba0d10d58f0f481c540c"
IMAGE_REFERENCE = f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"
HOME_ASSISTANT_VERSION = "2026.8.3"
PLATFORM = "linux/amd64"

RUN_LABEL = "io.github.iflytek.domux.ha-acceptance-run"
CONTAINER_PREFIX = "domux-ha-acceptance"
VOLUME_PREFIX = "domux-ha-acceptance-config"
CONTAINER_PORT = 8123
CPU_LIMIT = 1.5
NANO_CPUS = 1_500_000_000
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
PIDS_LIMIT = 512
PULL_TIMEOUT_SECONDS = 30 * 60
COMMAND_TIMEOUT_SECONDS = 60
READINESS_TIMEOUT_SECONDS = 240
ENTITY_TIMEOUT_SECONDS = 120
STATE_TIMEOUT_SECONDS = 30

ONBOARDING_STEPS = ("user", "core_config", "analytics", "integration")
ENTITY_IDS = {
    "light": "light.bed_light",
    "light_alternative": "light.ceiling_lights",
    "cover": "cover.hall_window",
    "climate": "climate.hvac",
}

CONFIGURATION_YAML = """homeassistant:
  name: Domux HA Acceptance
  latitude: 0
  longitude: 0
  elevation: 0
  unit_system: metric
  time_zone: UTC
  currency: USD

http:
api:
auth:
onboarding:
demo:

logger:
  default: warning
"""


class AcceptanceError(RuntimeError):
    """An expected, credential-safe acceptance failure."""


@dataclass(frozen=True)
class HttpResult:
    """A parsed HTTP result."""

    status: int
    payload: Any


@dataclass(frozen=True)
class PreparedRuntime:
    """Internal runtime details that must not be serialized as evidence."""

    base_url: str
    image: dict[str, Any]


@dataclass(frozen=True)
class SutCase:
    """One fixed Domux request and its expected server-side service shape."""

    name: str
    domain: str
    entity_id: str
    utterance: str
    raw_output: str
    clarification_answer: str | None
    confirmed_output: str | None
    expected_service: str
    expected_service_data: Mapping[str, object]
    expected_before: Mapping[str, object]
    expected_after: Mapping[str, object]


SUT_CASES = (
    SutCase(
        name="clarified_light_brightness",
        domain="light",
        entity_id="light.bed_light",
        utterance="Set the light brightness to 50 percent.",
        raw_output="set|Light|brightness|50|Percent|*|*",
        clarification_answer=(
            "Use the Bedroom Bed Light and set its brightness to 50 percent."
        ),
        confirmed_output=(
            "set|Bed Light|brightness|50|Percent|Bedroom|Ground Floor"
        ),
        expected_service="turn_on",
        expected_service_data={
            "brightness_pct": 50.0,
            "entity_id": "light.bed_light",
        },
        expected_before={"brightness": 64, "state": "on"},
        expected_after={"brightness": 128, "state": "on"},
    ),
    SutCase(
        name="unique_cover_position",
        domain="cover",
        entity_id="cover.hall_window",
        utterance="Set the Hall Window position to 20 percent.",
        raw_output="set|Hall Window|position|20|Percent|Hall|*",
        clarification_answer=None,
        confirmed_output=None,
        expected_service="set_cover_position",
        expected_service_data={
            "entity_id": "cover.hall_window",
            "position": 20,
        },
        expected_before={"current_position": 10, "state": "open"},
        expected_after={"current_position": 20, "state": "open"},
    ),
    SutCase(
        name="unique_climate_temperature",
        domain="climate",
        entity_id="climate.hvac",
        utterance="Set the Office HVAC temperature to 23 Celsius.",
        raw_output="set|Hvac|temperature|23|Celsius|Office|*",
        clarification_answer=None,
        confirmed_output=None,
        expected_service="set_temperature",
        expected_service_data={
            "entity_id": "climate.hvac",
            "temperature": 23.0,
        },
        expected_before={"state": "cool", "temperature": 21},
        expected_after={"state": "cool", "temperature": 23},
    ),
)


def sut_registry() -> EntityRegistry:
    """Return the fixed allow-list corresponding to pinned HA demo entities."""

    return EntityRegistry(
        (
            EntitySpec(
                "light.bed_light",
                "light",
                "Bed Light",
                "Bedroom",
                "Ground Floor",
                ("Bedroom Light",),
            ),
            EntitySpec(
                "light.ceiling_lights",
                "light",
                "Ceiling Lights",
                "Living Room",
                "Ground Floor",
                ("Living Room Lights",),
            ),
            EntitySpec(
                "cover.hall_window",
                "cover",
                "Hall Window",
                "Hall",
                "Ground Floor",
            ),
            EntitySpec(
                "climate.hvac",
                "climate",
                "Hvac",
                "Office",
                "Ground Floor",
                ("HVAC", "Office HVAC"),
            ),
        )
    )


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _parse_single_inspect(stdout: bytes | str, resource: str) -> dict[str, Any]:
    try:
        payload = json.loads(_decode(stdout))
    except json.JSONDecodeError as exc:
        raise AcceptanceError(f"docker {resource} inspect returned invalid JSON") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise AcceptanceError(f"docker {resource} inspect returned an unexpected shape")
    return payload[0]


def configuration_archive() -> bytes:
    """Build a deterministic in-memory tar archive for ``docker cp -``."""

    content = CONFIGURATION_YAML.encode("utf-8")
    info = tarfile.TarInfo("configuration.yaml")
    info.size = len(content)
    info.mode = 0o600
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class DockerCli:
    """A minimal Docker CLI adapter with ownership-checked cleanup."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        run_id: str | None = None,
    ) -> None:
        self._runner = runner
        self.run_id = run_id or secrets.token_hex(8)
        if not re.fullmatch(r"[a-f0-9]{12,64}", self.run_id):
            raise ValueError("run_id must be 12-64 lowercase hexadecimal characters")
        self.container_name = f"{CONTAINER_PREFIX}-{self.run_id}"
        self.volume_name = f"{VOLUME_PREFIX}-{self.run_id}"
        self._container_may_exist = False
        self._volume_may_exist = False

    def _invoke(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[bytes]:
        command = ["docker", *arguments]
        try:
            return self._runner(
                command,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            operation = arguments[0] if arguments else "command"
            raise AcceptanceError(f"docker {operation} timed out") from None
        except OSError:
            raise AcceptanceError("docker CLI could not be executed") from None

    def _run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        completed = self._invoke(arguments, input_bytes=input_bytes, timeout=timeout)
        if completed.returncode != 0:
            operation = arguments[0] if arguments else "command"
            raise AcceptanceError(f"docker {operation} failed")
        return _decode(completed.stdout)

    def _optional_inspect(self, resource: str, name: str) -> dict[str, Any] | None:
        completed = self._invoke([resource, "inspect", name])
        if completed.returncode == 0:
            return _parse_single_inspect(completed.stdout, resource)
        stderr = _decode(completed.stderr).lower()
        if completed.returncode == 1 and (
            "no such" in stderr or "not found" in stderr
        ):
            return None
        raise AcceptanceError(f"docker {resource} inspect failed")

    def _validate_volume_ownership(self, inspect: Mapping[str, Any]) -> None:
        labels = inspect.get("Labels")
        if not isinstance(labels, dict) or labels.get(RUN_LABEL) != self.run_id:
            raise AcceptanceError("refusing to manage a volume without the task run label")
        if inspect.get("Name") != self.volume_name:
            raise AcceptanceError("volume identity does not match the task resource")

    def _validate_container_ownership(self, inspect: Mapping[str, Any]) -> None:
        config = inspect.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if not isinstance(labels, dict) or labels.get(RUN_LABEL) != self.run_id:
            raise AcceptanceError("refusing to manage a container without the task run label")
        name = inspect.get("Name")
        if name not in {self.container_name, f"/{self.container_name}"}:
            raise AcceptanceError("container identity does not match the task resource")

    def _inspect_image(self) -> dict[str, Any]:
        raw = self._run(["image", "inspect", IMAGE_REFERENCE])
        inspect = _parse_single_inspect(raw, "image")
        labels = inspect.get("Config", {}).get("Labels", {})
        repo_digests = inspect.get("RepoDigests", [])
        if inspect.get("Os") != "linux" or inspect.get("Architecture") != "amd64":
            raise AcceptanceError("pinned image does not resolve to linux/amd64")
        if not isinstance(labels, dict) or labels.get("io.hass.version") != HOME_ASSISTANT_VERSION:
            raise AcceptanceError("pinned image Home Assistant version label is unexpected")
        if not isinstance(repo_digests, list) or IMAGE_REFERENCE not in repo_digests:
            raise AcceptanceError("official repository digest is absent after pull")
        if inspect.get("Config", {}).get("Healthcheck") is not None:
            raise AcceptanceError("pinned image unexpectedly defines a Docker healthcheck")
        return {
            "architecture": "amd64",
            "docker_healthcheck": False,
            "manifest_digest": IMAGE_DIGEST,
            "operating_system": "linux",
            "repository": IMAGE_REPOSITORY,
            "version": HOME_ASSISTANT_VERSION,
        }

    def _validate_runtime(self, inspect: Mapping[str, Any]) -> None:
        self._validate_container_ownership(inspect)
        config = inspect.get("Config", {})
        host = inspect.get("HostConfig", {})
        if not isinstance(config, dict) or config.get("Image") != IMAGE_REFERENCE:
            raise AcceptanceError("container does not use the pinned official image")
        if not isinstance(host, dict):
            raise AcceptanceError("container host configuration is missing")
        restart = host.get("RestartPolicy", {})
        if not isinstance(restart, dict) or restart.get("Name") != "no":
            raise AcceptanceError("container restart policy is not disabled")
        if host.get("NanoCpus") != NANO_CPUS:
            raise AcceptanceError("container CPU limit differs from the acceptance profile")
        if host.get("Memory") != MEMORY_LIMIT_BYTES:
            raise AcceptanceError("container memory limit differs from the acceptance profile")
        if host.get("PidsLimit") != PIDS_LIMIT:
            raise AcceptanceError("container PID limit differs from the acceptance profile")
        if host.get("Privileged") is not False:
            raise AcceptanceError("container must not be privileged")
        if host.get("NetworkMode") not in {"bridge", "default"}:
            raise AcceptanceError("container must use an isolated bridge network")

        port_bindings = host.get("PortBindings", {})
        expected_key = f"{CONTAINER_PORT}/tcp"
        if not isinstance(port_bindings, dict) or set(port_bindings) != {expected_key}:
            raise AcceptanceError("container has unexpected published ports")
        bindings = port_bindings[expected_key]
        if (
            not isinstance(bindings, list)
            or len(bindings) != 1
            or bindings[0].get("HostIp") != "127.0.0.1"
            or bindings[0].get("HostPort") not in {"", "0"}
        ):
            raise AcceptanceError("container port is not a random loopback-only binding")

        mounts = inspect.get("Mounts")
        if not isinstance(mounts, list) or len(mounts) != 1:
            raise AcceptanceError("container must have exactly one mount")
        mount = mounts[0]
        if (
            not isinstance(mount, dict)
            or mount.get("Type") != "volume"
            or mount.get("Name") != self.volume_name
            or mount.get("Destination") != "/config"
            or mount.get("RW") is not True
        ):
            raise AcceptanceError("container config mount is not the task named volume")

    def prepare(self) -> PreparedRuntime:
        """Pull, validate, create, configure, and start the isolated runtime."""

        self._run(
            ["pull", "--platform", PLATFORM, IMAGE_REFERENCE],
            timeout=PULL_TIMEOUT_SECONDS,
        )
        image = self._inspect_image()

        self._volume_may_exist = True
        created_volume = self._run(
            [
                "volume",
                "create",
                "--label",
                f"{RUN_LABEL}={self.run_id}",
                self.volume_name,
            ]
        ).strip()
        if created_volume != self.volume_name:
            raise AcceptanceError("docker returned an unexpected volume identity")
        volume_inspect = self._optional_inspect("volume", self.volume_name)
        if volume_inspect is None:
            raise AcceptanceError("task named volume disappeared after creation")
        self._validate_volume_ownership(volume_inspect)

        self._container_may_exist = True
        self._run(
            [
                "create",
                "--name",
                self.container_name,
                "--label",
                f"{RUN_LABEL}={self.run_id}",
                "--restart=no",
                f"--cpus={CPU_LIMIT}",
                "--memory=2g",
                f"--pids-limit={PIDS_LIMIT}",
                "--env",
                "TZ=UTC",
                "--mount",
                f"type=volume,source={self.volume_name},target=/config",
                "--publish",
                f"127.0.0.1::{CONTAINER_PORT}",
                IMAGE_REFERENCE,
            ]
        )
        container_inspect = self._optional_inspect("container", self.container_name)
        if container_inspect is None:
            raise AcceptanceError("task container disappeared after creation")
        self._validate_runtime(container_inspect)

        self._run(
            ["cp", "-", f"{self.container_name}:/config"],
            input_bytes=configuration_archive(),
        )
        self._run(["start", self.container_name])
        binding = self._run(
            ["port", self.container_name, f"{CONTAINER_PORT}/tcp"]
        ).strip()
        match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", binding)
        if match is None:
            raise AcceptanceError("Docker returned a non-loopback or malformed port binding")
        host_port = int(match.group(1))
        if not 1024 <= host_port <= 65535:
            raise AcceptanceError("Docker-assigned host port is not a high port")
        return PreparedRuntime(
            base_url=f"http://127.0.0.1:{host_port}",
            image=image,
        )

    def cleanup(self) -> None:
        """Remove only this run's label-verified container and named volume."""

        failures: list[str] = []
        if self._container_may_exist:
            try:
                inspect = self._optional_inspect("container", self.container_name)
                if inspect is not None:
                    self._validate_container_ownership(inspect)
                    self._run(["rm", "--force", self.container_name])
                if self._optional_inspect("container", self.container_name) is not None:
                    raise AcceptanceError("task container still exists after removal")
                self._container_may_exist = False
            except AcceptanceError:
                failures.append("container")

        if self._volume_may_exist:
            try:
                inspect = self._optional_inspect("volume", self.volume_name)
                if inspect is not None:
                    self._validate_volume_ownership(inspect)
                    self._run(["volume", "rm", self.volume_name])
                if self._optional_inspect("volume", self.volume_name) is not None:
                    raise AcceptanceError("task volume still exists after removal")
                self._volume_may_exist = False
            except AcceptanceError:
                failures.append("volume")

        if failures:
            resources = " and ".join(failures)
            raise AcceptanceError(f"failed to clean the task-owned {resources}")


class HomeAssistantApi:
    """Small Home Assistant HTTP REST/onboarding/auth client."""

    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., Any] = urlopen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._opener = opener
        self._monotonic = monotonic
        self._sleep = sleep

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        form_body: Mapping[str, str] | None = None,
        token: str | None = None,
        expected: Iterable[int] = (200,),
        timeout: float = 30,
    ) -> HttpResult:
        """Call one HA endpoint without including request data in errors."""

        if json_body is not None and form_body is not None:
            raise ValueError("request cannot contain both JSON and form data")
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif form_body is not None:
            body = urlencode(form_body).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(request, timeout=timeout) as response:
                status = response.status
                raw = response.read()
        except HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except (URLError, TimeoutError, OSError):
            raise AcceptanceError(f"{method} {path} failed") from None

        payload: Any = None
        if raw:
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = raw.decode("utf-8", errors="replace")
        expected_statuses = set(expected)
        if status not in expected_statuses:
            raise AcceptanceError(f"{method} {path} returned unexpected HTTP {status}")
        return HttpResult(status=status, payload=payload)

    def wait_for_readiness(self) -> HttpResult:
        deadline = self._monotonic() + READINESS_TIMEOUT_SECONDS
        while True:
            try:
                result = self.request(
                    "GET", "/api/onboarding", expected=(200,), timeout=5
                )
                if isinstance(result.payload, list):
                    return result
            except AcceptanceError:
                pass
            if self._monotonic() >= deadline:
                raise AcceptanceError("Home Assistant onboarding endpoint was not ready")
            self._sleep(1)

    def wait_for_entities(self, entity_ids: Iterable[str], token: str) -> None:
        expected = set(entity_ids)
        deadline = self._monotonic() + ENTITY_TIMEOUT_SECONDS
        while True:
            result = self.request("GET", "/api/states", token=token)
            if isinstance(result.payload, list):
                actual = {
                    item.get("entity_id")
                    for item in result.payload
                    if isinstance(item, dict)
                }
                if expected <= actual:
                    return
            if self._monotonic() >= deadline:
                raise AcceptanceError("required Home Assistant demo entities were not loaded")
            self._sleep(1)

    def wait_for_state(
        self, entity_id: str, desired: str, token: str
    ) -> dict[str, Any]:
        deadline = self._monotonic() + STATE_TIMEOUT_SECONDS
        while True:
            result = self.request(
                "GET", f"/api/states/{entity_id}", token=token
            )
            if isinstance(result.payload, dict) and result.payload.get("state") == desired:
                return result.payload
            if self._monotonic() >= deadline:
                raise AcceptanceError(f"{entity_id} did not reach the expected state")
            self._sleep(0.25)

    def wait_for_setup_projection(
        self,
        entity_id: str,
        token: str,
        *,
        state: str,
        attributes: Mapping[str, object],
    ) -> dict[str, Any]:
        """Wait only for direct-REST setup fields, never for a SUT outcome."""

        deadline = self._monotonic() + STATE_TIMEOUT_SECONDS
        while True:
            result = self.request(
                "GET", f"/api/states/{entity_id}", token=token
            )
            if isinstance(result.payload, dict):
                actual_attributes = result.payload.get("attributes")
                if isinstance(actual_attributes, Mapping):
                    attributes_match = all(
                        _scalar_matches(actual_attributes.get(key), expected)
                        for key, expected in attributes.items()
                    )
                    if result.payload.get("state") == state and attributes_match:
                        return result.payload
            if self._monotonic() >= deadline:
                raise AcceptanceError(
                    f"{entity_id} did not reach its deterministic setup projection"
                )
            self._sleep(0.25)


def _onboarding_state(payload: Any, expected_done: bool) -> dict[str, bool]:
    if not isinstance(payload, list):
        raise AcceptanceError("onboarding response is not a list")
    states: dict[str, bool] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise AcceptanceError("onboarding response contains a non-object")
        step = item.get("step")
        done = item.get("done")
        if step not in ONBOARDING_STEPS or not isinstance(done, bool) or step in states:
            raise AcceptanceError("onboarding response contains unexpected step data")
        states[step] = done
    if set(states) != set(ONBOARDING_STEPS):
        raise AcceptanceError("onboarding response does not contain the four pinned steps")
    if any(states[step] is not expected_done for step in ONBOARDING_STEPS):
        state_name = "complete" if expected_done else "fresh"
        raise AcceptanceError(f"onboarding state is not {state_name}")
    return {step: states[step] for step in ONBOARDING_STEPS}


def _service_call(
    api: HomeAssistantApi,
    domain: str,
    service: str,
    entity_id: str,
    token: str,
    extra: Mapping[str, Any] | None = None,
) -> int:
    body: dict[str, Any] = {"entity_id": entity_id}
    if extra:
        body.update(extra)
    return api.request(
        "POST",
        f"/api/services/{domain}/{service}",
        json_body=body,
        token=token,
    ).status


def _scalar_matches(actual: object, expected: object) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=0.01)
    return actual == expected


def _assert_projection_subset(
    actual: Mapping[str, object] | None,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if actual is None or any(
        key not in actual or not _scalar_matches(actual[key], value)
        for key, value in expected.items()
    ):
        raise AcceptanceError(f"{label} does not match the pinned controlled projection")


def normalize_setup_state(
    api: HomeAssistantApi, access_token: str
) -> dict[str, Any]:
    """Normalize deterministic demo state using direct REST outside the SUT."""

    calls: list[dict[str, object]] = []

    def setup_call(
        domain: str,
        service: str,
        entity_id: str,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        status = _service_call(
            api,
            domain,
            service,
            entity_id,
            access_token,
            extra=extra,
        )
        calls.append({"domain": domain, "http": status, "service": service})

    setup_call(
        "light",
        "turn_on",
        ENTITY_IDS["light"],
        {"brightness_pct": 25, "color_temp_kelvin": 3000},
    )
    setup_call(
        "cover",
        "set_cover_position",
        ENTITY_IDS["cover"],
        {"position": 10},
    )
    setup_call(
        "climate",
        "set_hvac_mode",
        ENTITY_IDS["climate"],
        {"hvac_mode": "cool"},
    )
    setup_call(
        "climate",
        "set_temperature",
        ENTITY_IDS["climate"],
        {"temperature": 21},
    )

    api.wait_for_setup_projection(
        ENTITY_IDS["light"],
        access_token,
        state="on",
        attributes={"brightness": 64, "color_temp_kelvin": 3000},
    )
    api.wait_for_setup_projection(
        ENTITY_IDS["cover"],
        access_token,
        state="open",
        attributes={"current_position": 10},
    )
    api.wait_for_setup_projection(
        ENTITY_IDS["climate"],
        access_token,
        state="cool",
        attributes={"temperature": 21},
    )
    return {
        "classification": "direct_rest_state_normalization",
        "dispatches": calls,
        "included_in_sut_dispatch_count": False,
        "purpose": "setup_only",
    }


def create_rest_adapter(base_url: str, token: str) -> HomeAssistantRESTAdapter:
    """Create the real SUT adapter with bounded loopback polling."""

    return HomeAssistantRESTAdapter(
        base_url,
        token,
        timeout_seconds=10,
        poll_seconds=15,
    )


def _confirmed_instruction(case: SutCase) -> DomuxInstruction | None:
    if case.confirmed_output is None:
        return None
    parsed = parse_domux_output(case.confirmed_output)
    if len(parsed) != 1:
        raise AcceptanceError("SUT confirmation must contain exactly one instruction")
    return parsed[0]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{label} is not an object")
    return dict(value)


def run_sut_cases(
    adapter: HomeAssistantRESTAdapter,
) -> dict[str, Any]:
    """Run grounding through one-time commit against the real REST adapter."""

    if not isinstance(adapter, HomeAssistantRESTAdapter):
        raise AcceptanceError("SUT adapter must be HomeAssistantRESTAdapter")
    if adapter.sut_calls:
        raise AcceptanceError("SUT adapter already contains dispatch history")

    registry = sut_registry()
    store = PreparedActionStore(ttl_seconds=30)
    reports: list[dict[str, Any]] = []
    for case in SUT_CASES:
        grounded = ground_domux_request(
            case.utterance,
            case.raw_output,
            registry,
        )
        confirmed = _confirmed_instruction(case)
        if case.clarification_answer is None:
            if confirmed is not None:
                raise AcceptanceError("unique SUT case unexpectedly has a confirmation")
            resolved = resolve_unique_request(grounded, registry)
            resolution = "resolve_unique_request"
        else:
            if confirmed is None:
                raise AcceptanceError("clarified SUT case is missing its confirmation")
            resolved = resolve_clarification_submission(
                grounded,
                answer=case.clarification_answer,
                confirmed_instruction=confirmed,
                registry=registry,
            )
            resolution = "resolve_clarification_submission"
        if resolved.chosen.entity_id != case.entity_id:
            raise AcceptanceError("grounding resolved to an unexpected entity")

        prepared = store.prepare(
            actor_id="ha-acceptance-actor",
            session_id="ha-acceptance-session",
            grounded=grounded,
            registry=registry,
            adapter=adapter,
            clarification_answer=case.clarification_answer,
            confirmed_instruction=confirmed,
        )
        snapshot = store.snapshot(prepared.nonce)
        plan = _mapping(snapshot.get("plan"), "prepared plan")
        service_data = _mapping(plan.get("service_data"), "prepared service data")
        expected_projection = _mapping(
            plan.get("expected_projection"), "prepared expected projection"
        )
        if (
            prepared.entity_id != case.entity_id
            or plan.get("entity_id") != case.entity_id
            or plan.get("domain") != case.domain
            or plan.get("service") != case.expected_service
            or service_data != dict(case.expected_service_data)
        ):
            raise AcceptanceError("prepared action has an unexpected service shape")

        dispatch_count_before = len(adapter.sut_calls)
        committed = store.commit(
            prepared.confirmation(),
            registry=registry,
            adapter=adapter,
        )
        if (
            not committed.accepted
            or not committed.dispatched
            or not committed.acknowledged
            or committed.outcome_unknown
            or committed.reason != "committed"
            or committed.status != "COMMITTED"
            or committed.before_registry_digest is None
            or committed.after_registry_digest is None
        ):
            raise AcceptanceError(
                f"SUT case {case.name} did not commit: "
                f"{committed.status}/{committed.reason}"
            )
        if len(adapter.sut_calls) != dispatch_count_before + 1:
            raise AcceptanceError("commit did not produce exactly one SUT dispatch")
        event = adapter.sut_calls[-1]
        if (
            event.get("kind") != "sut"
            or event.get("domain") != case.domain
            or event.get("service") != case.expected_service
            or event.get("data") != dict(case.expected_service_data)
            or event.get("acknowledged") is not True
            or event.get("outcome") != "observed"
        ):
            raise AcceptanceError("HomeAssistantRESTAdapter dispatch evidence is unexpected")

        controlled_before = _mapping(committed.before, "controlled before state")
        controlled_after = _mapping(committed.after, "controlled after state")
        _assert_projection_subset(
            controlled_before,
            case.expected_before,
            label="controlled before state",
        )
        _assert_projection_subset(
            controlled_after,
            case.expected_after,
            label="controlled after state",
        )
        if not projection_matches(controlled_after, expected_projection):
            raise AcceptanceError("committed state does not match the prepared postcondition")

        dispatch_count_before_replay = len(adapter.sut_calls)
        replay = store.commit(
            prepared.confirmation(),
            registry=registry,
            adapter=adapter,
        )
        replay_dispatch_delta = len(adapter.sut_calls) - dispatch_count_before_replay
        if (
            replay.accepted
            or replay.dispatched
            or replay.reason != "replayed_nonce"
            or replay_dispatch_delta != 0
        ):
            raise AcceptanceError("nonce replay was not rejected with zero dispatch")

        reports.append(
            {
                "case": case.name,
                "controlled_after": controlled_after,
                "controlled_before": controlled_before,
                "domain": case.domain,
                "grounding": {
                    "candidate_ids": [
                        candidate.entity_id for candidate in grounded.candidates
                    ],
                    "clarification_required": grounded.clarification.required,
                    "resolution": resolution,
                    "selected_entity_id": resolved.chosen.entity_id,
                },
                "postcondition": {
                    "all_registered_entities_exact": True,
                    "matched_prepared_projection": True,
                    "reason": committed.reason,
                    "status": committed.status,
                },
                "replay": {
                    "accepted": replay.accepted,
                    "dispatched": replay.dispatched,
                    "reason": replay.reason,
                    "sut_dispatch_delta": replay_dispatch_delta,
                },
                "service_shape": {
                    "data": service_data,
                    "domain": case.domain,
                    "service": case.expected_service,
                },
            }
        )

    if len(adapter.sut_calls) != len(SUT_CASES):
        raise AcceptanceError("SUT dispatch count differs from the fixed case count")
    return {
        "adapter": "HomeAssistantRESTAdapter",
        "cases": reports,
        "classification": "clarify_commit_sut",
        "pipeline": [
            "ground_domux_request",
            "resolve_clarification_submission_or_unique",
            "PreparedActionStore.prepare",
            "PreparedActionStore.commit",
            "HomeAssistantRESTAdapter.call_service",
        ],
        "sut_dispatch_total": len(adapter.sut_calls),
    }


def generate_credentials() -> tuple[str, str]:
    """Generate one synthetic local-only account."""

    return f"acceptance_{secrets.token_hex(8)}", secrets.token_urlsafe(32)


def exercise_home_assistant(
    api: HomeAssistantApi,
    *,
    credential_factory: Callable[[], tuple[str, str]] = generate_credentials,
    rest_adapter_factory: Callable[
        [str, str], HomeAssistantRESTAdapter
    ] = create_rest_adapter,
) -> dict[str, Any]:
    """Complete onboarding, execute the real SUT pipeline, and revoke auth."""

    readiness = api.wait_for_readiness()
    initial_steps = _onboarding_state(readiness.payload, expected_done=False)
    unauthenticated = api.request("GET", "/api/", expected=(401,))

    username, password = credential_factory()
    if not username or not password:
        raise AcceptanceError("credential factory returned an empty value")
    client_id = f"{api.base_url}/"
    users = api.request(
        "POST",
        "/api/onboarding/users",
        json_body={
            "name": "Domux Acceptance",
            "username": username,
            "password": password,
            "client_id": client_id,
            "language": "en",
        },
    )
    if not isinstance(users.payload, dict) or not isinstance(
        users.payload.get("auth_code"), str
    ):
        raise AcceptanceError("user onboarding response omitted its auth code")
    user_auth_code = users.payload["auth_code"]

    issued = api.request(
        "POST",
        "/auth/token",
        form_body={
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": user_auth_code,
        },
    )
    if not isinstance(issued.payload, dict):
        raise AcceptanceError("token endpoint returned a non-object")
    access_token = issued.payload.get("access_token")
    refresh_token = issued.payload.get("refresh_token")
    token_type = issued.payload.get("token_type")
    expires_in = issued.payload.get("expires_in")
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        raise AcceptanceError("token endpoint omitted access or refresh token")
    if token_type != "Bearer" or expires_in != 1800:
        raise AcceptanceError("token endpoint returned an unexpected type or lifetime")

    authenticated = api.request("GET", "/api/", token=access_token)
    if not isinstance(authenticated.payload, dict) or authenticated.payload.get(
        "message"
    ) != "API running.":
        raise AcceptanceError("authenticated API health response is unexpected")

    core_config = api.request(
        "POST", "/api/onboarding/core_config", token=access_token, timeout=60
    )
    integration = api.request(
        "POST",
        "/api/onboarding/integration",
        json_body={
            "client_id": client_id,
            "redirect_uri": f"{api.base_url}/auth-callback",
        },
        token=access_token,
    )
    if not isinstance(integration.payload, dict) or not isinstance(
        integration.payload.get("auth_code"), str
    ):
        raise AcceptanceError("integration onboarding response omitted its auth code")
    integration_auth_code = integration.payload["auth_code"]
    analytics = api.request(
        "POST", "/api/onboarding/analytics", token=access_token
    )
    final = api.request("GET", "/api/onboarding")
    final_steps = _onboarding_state(final.payload, expected_done=True)

    api.wait_for_entities(ENTITY_IDS.values(), access_token)
    setup = normalize_setup_state(api, access_token)
    adapter = rest_adapter_factory(api.base_url, access_token)
    sut = run_sut_cases(adapter)

    revoked = api.request(
        "POST", "/auth/revoke", form_body={"token": refresh_token}
    )
    refresh_after_revoke = api.request(
        "POST",
        "/auth/token",
        form_body={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        expected=(400,),
    )

    result = {
        "auth": {
            "issue_http": issued.status,
            "refresh_after_revoke_http": refresh_after_revoke.status,
            "revoke_http": revoked.status,
            "token_type": token_type,
            "ttl_seconds": expires_in,
        },
        "health": {
            "authenticated_api_http": authenticated.status,
            "message": authenticated.payload["message"],
            "unauthenticated_api_http": unauthenticated.status,
        },
        "onboarding": {
            "final": final_steps,
            "initial": initial_steps,
            "requests": {
                "analytics_http": analytics.status,
                "core_config_http": core_config.status,
                "integration_http": integration.status,
                "users_http": users.status,
            },
        },
        "readiness": {
            "endpoint": "/api/onboarding",
            "http": readiness.status,
        },
        "phases": {
            "setup": setup,
            "sut": sut,
            "teardown": {
                "classification": "credential_cleanup",
                "refresh_after_revoke_http": refresh_after_revoke.status,
                "refresh_revoke_http": revoked.status,
            },
        },
    }
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    forbidden = (
        username,
        password,
        user_auth_code,
        integration_auth_code,
        access_token,
        refresh_token,
        client_id,
        api.base_url,
    )
    if any(value and value in serialized for value in forbidden):
        raise AcceptanceError("acceptance result contains private runtime material")
    return result


def execute_acceptance(
    docker: DockerCli,
    *,
    api_factory: Callable[[str], HomeAssistantApi] = HomeAssistantApi,
    credential_factory: Callable[[], tuple[str, str]] = generate_credentials,
    rest_adapter_factory: Callable[
        [str, str], HomeAssistantRESTAdapter
    ] = create_rest_adapter,
) -> dict[str, Any]:
    """Run acceptance and always clean exactly the resources owned by this run."""

    try:
        runtime = docker.prepare()
        home_assistant = exercise_home_assistant(
            api_factory(runtime.base_url),
            credential_factory=credential_factory,
            rest_adapter_factory=rest_adapter_factory,
        )
        return {
            "home_assistant": home_assistant,
            "image": runtime.image,
            "isolation": {
                "container_count": 1,
                "cpu_limit": CPU_LIMIT,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "named_volume_count": 1,
                "pids_limit": PIDS_LIMIT,
                "random_loopback_binding": True,
                "restart_policy": "no",
            },
            "schema_version": 1,
            "status": "passed",
        }
    finally:
        docker.cleanup()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def atomic_write_json(output: Path, value: object) -> bytes:
    """Atomically publish one deterministic JSON artifact with mode 0600."""

    payload = canonical_json_bytes(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned official Home Assistant acceptance check."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination for the deterministic redacted JSON result",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    docker_factory: Callable[[], DockerCli] = DockerCli,
    api_factory: Callable[[str], HomeAssistantApi] = HomeAssistantApi,
    credential_factory: Callable[[], tuple[str, str]] = generate_credentials,
    rest_adapter_factory: Callable[
        [str, str], HomeAssistantRESTAdapter
    ] = create_rest_adapter,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_acceptance(
            docker_factory(),
            api_factory=api_factory,
            credential_factory=credential_factory,
            rest_adapter_factory=rest_adapter_factory,
        )
        payload = atomic_write_json(args.output, result)
    except (AcceptanceError, OSError, ValueError) as exc:
        print(f"ha_acceptance failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("ha_acceptance failed: unexpected internal error", file=sys.stderr)
        return 1
    sys.stdout.write(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
