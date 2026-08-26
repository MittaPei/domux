from __future__ import annotations

import json
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from clarify_commit import (  # noqa: E402
    AdapterError,
    ClarifyPrepareStore,
    DomuxInstruction,
    EntityRegistry,
    EntitySpec,
    GroundingError,
    HomeAssistantRESTAdapter,
    InMemoryHAAdapter,
    ParseError,
    PreparedActionStore,
    ServiceCallError,
    ServiceCallResult,
    SessionContext,
    altered_confirmation,
    build_plan,
    clarification_for,
    controlled_projection,
    ground_domux_request,
    parse_domux_output,
    planning_projection,
    projection_matches,
    resolve_clarification,
    resolve_clarification_submission,
)


def fixture() -> tuple[EntityRegistry, dict[str, dict[str, object]]]:
    entities = (
        EntitySpec("light.living_ceiling", "light", "Ceiling Light", "Living Room", "Ground Floor"),
        EntitySpec("light.study_ceiling", "light", "Ceiling Light", "Study", "Ground Floor"),
        EntitySpec("light.utility", "light", "Utility Light", "Utility Room", "Ground Floor"),
        EntitySpec("cover.study_curtain", "cover", "Curtain", "Study", "Ground Floor"),
        EntitySpec("climate.study_ac", "climate", "AC", "Study", "Ground Floor"),
    )
    states = {
        "light.living_ceiling": {
            "entity_id": "light.living_ceiling", "state": "on", "attributes": {
                "brightness": 204, "supported_color_modes": ["brightness", "color_temp", "rgb"],
                "min_color_temp_kelvin": 3000, "max_color_temp_kelvin": 6500,
            },
        },
        "light.study_ceiling": {
            "entity_id": "light.study_ceiling", "state": "on", "attributes": {
                "brightness": 153, "supported_color_modes": ["brightness", "color_temp", "rgb"],
                "min_color_temp_kelvin": 3000, "max_color_temp_kelvin": 6500,
            },
        },
        "light.utility": {
            "entity_id": "light.utility", "state": "off", "attributes": {
                "brightness": 0, "supported_color_modes": ["brightness", "color_temp", "rgb"],
                "min_color_temp_kelvin": 3000, "max_color_temp_kelvin": 6500,
            },
        },
        "cover.study_curtain": {
            "entity_id": "cover.study_curtain", "state": "open",
            "attributes": {"current_position": 80, "supported_features": 7},
        },
        "climate.study_ac": {
            "entity_id": "climate.study_ac", "state": "cool",
            "attributes": {
                "temperature": 24.0, "fan_mode": "medium",
                "hvac_modes": ["off", "cool", "heat", "dry", "fan_only", "auto"],
                "fan_modes": ["low", "medium", "medium_high", "high"],
                "supported_features": 9, "temperature_unit": "°C",
                "min_temp": 16.0, "max_temp": 30.0, "target_temp_step": 0.5,
            },
        },
    }
    return EntityRegistry(entities), states


class MutableClock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class ParserTests(unittest.TestCase):
    def test_single_and_multiple_instructions(self) -> None:
        one = parse_domux_output("turnOff|Ceiling Light|*|*|*|Study|Ground Floor")
        self.assertEqual(one[0].room, "Study")
        many = parse_domux_output(
            "turnOn|Light|*|*|*|Study|*\nset|Light|brightness|50|Percent|Study|*"
        )
        self.assertEqual(len(many), 2)

    def test_ampersand_is_an_explicit_multi_instruction_separator(self) -> None:
        parsed = parse_domux_output(
            "turnOn|Light|*|*|*|Study|*&turnOff|Light|*|*|*|Bedroom|*"
        )
        self.assertEqual(len(parsed), 2)

    def test_malformed_raw_outputs_fail_closed(self) -> None:
        invalid = ("", "hello", "turnOn|Light", "*|Light|*|*|*|Study|*", "turnOn||*|*|*|Study|*")
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ParseError):
                parse_domux_output(raw)

    def test_raw_text_is_not_rewritten(self) -> None:
        raw = " turnOff | Ceiling Light | * | * | * | Study | Ground Floor "
        parsed = parse_domux_output(raw)
        self.assertEqual(parsed[0].device, "Ceiling Light")
        self.assertEqual(raw, " turnOff | Ceiling Light | * | * | * | Study | Ground Floor ")


class GroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry, self.states = fixture()

    def test_explicit_room_is_unique(self) -> None:
        instruction = parse_domux_output("turnOff|Ceiling Light|*|*|*|Study|Ground Floor")[0]
        candidates = self.registry.candidates(instruction)
        self.assertEqual([item.entity_id for item in candidates], ["light.study_ceiling"])
        self.assertFalse(clarification_for(candidates).required)

    def test_omitted_room_is_ambiguous_and_ordered(self) -> None:
        instruction = parse_domux_output("turnOff|Ceiling Light|*|*|*|*|*")[0]
        candidates = self.registry.candidates(instruction)
        self.assertEqual(
            [item.entity_id for item in candidates],
            ["light.living_ceiling", "light.study_ceiling"],
        )
        prompt = clarification_for(candidates)
        self.assertTrue(prompt.required)
        self.assertLessEqual(len(prompt.candidates), 3)
        self.assertEqual(resolve_clarification("Study", prompt.candidates).entity_id, "light.study_ceiling")

    def test_duplicate_human_labels_remain_visibly_distinguishable(self) -> None:
        registry = EntityRegistry((
            EntitySpec(
                "light.study_a", "light", "Ceiling Light", "Study", "Ground Floor",
                ("North circuit",),
            ),
            EntitySpec(
                "light.study_b", "light", "Ceiling Light", "Study", "Ground Floor",
                ("South circuit",),
            ),
        ))
        grounded = ground_domux_request(
            "Turn off the ceiling light.",
            "turnOff|Ceiling Light|*|*|*|*|*",
            registry,
        )
        prompt = grounded.clarification.prompt
        self.assertIsNotNone(prompt)
        self.assertIn("alias: North circuit", prompt)
        self.assertIn("alias: South circuit", prompt)
        self.assertIn("id: light.study_a", prompt)
        self.assertIn("id: light.study_b", prompt)
        self.assertEqual(
            resolve_clarification("light.study_b", grounded.candidates).entity_id,
            "light.study_b",
        )

    def test_context_limits_pronoun_candidates_but_does_not_guess(self) -> None:
        instruction = parse_domux_output("turnOff|*|*|*|*|*|*")[0]
        context = SessionContext(("light.living_ceiling", "light.study_ceiling"))
        candidates = self.registry.candidates(instruction, context)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(clarification_for(candidates).required)

    def test_zero_and_ambiguous_answers_fail(self) -> None:
        candidates = self.registry.candidates(parse_domux_output("turnOff|Ceiling Light|*|*|*|*|*")[0])
        for answer in ("", "Kitchen", "Ceiling Light", "9"):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification(answer, candidates)

    def test_plan_mappings_cover_all_three_domains(self) -> None:
        adapter = InMemoryHAAdapter(self.states)
        cases = (
            ("turnOff|Ceiling Light|*|*|*|Study|Ground Floor", "light.study_ceiling", "turn_off"),
            ("set|Curtain|position|30|Percent|Study|Ground Floor", "cover.study_curtain", "set_cover_position"),
            ("set|AC|temperature|23|Celsius|Study|Ground Floor", "climate.study_ac", "set_temperature"),
        )
        for raw, entity_id, service in cases:
            entity = self.registry.get(entity_id)
            plan = build_plan(parse_domux_output(raw)[0], entity, adapter.get_state(entity_id))
            self.assertEqual(plan.service, service)

    def test_light_brightness_zero_uses_home_assistant_turn_off_semantics(self) -> None:
        entity = self.registry.get("light.study_ceiling")
        adapter = InMemoryHAAdapter(self.states)
        cases = (
            "set|Light|brightness|0|Percent|Study|Ground Floor",
            "adjustDown|Light|brightness|60|Percent|Study|Ground Floor",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                before = adapter.get_state(entity.entity_id)
                before["attributes"]["brightness"] = 153
                before["state"] = "on"
                adapter.set_state_for_setup(entity.entity_id, before)
                plan = build_plan(parse_domux_output(raw)[0], entity, before)
                self.assertEqual(plan.service, "turn_on")
                self.assertEqual(plan.service_data["brightness_pct"], 0)
                self.assertEqual(plan.expected_projection["state"], "off")
                self.assertIsNone(plan.expected_projection["brightness"])
                result = adapter.call_service(plan.domain, plan.service, plan.service_data)
                self.assertEqual(controlled_projection(result.after, "light")["state"], "off")
                self.assertTrue(projection_matches(
                    controlled_projection(result.after, "light"),
                    plan.expected_projection,
                ))

        positive = build_plan(
            parse_domux_output(
                "set|Light|brightness|1|Percent|Study|Ground Floor"
            )[0],
            entity,
            adapter.get_state(entity.entity_id),
        )
        self.assertEqual(positive.expected_projection["state"], "on")
        self.assertEqual(positive.expected_projection["brightness"], 3)

    def test_home_assistant_integer_coercions_and_optional_cover_position_are_bound(self) -> None:
        cover = self.registry.get("cover.study_curtain")
        cover_state = json.loads(json.dumps(self.states[cover.entity_id]))
        integral = build_plan(
            parse_domux_output(
                "set|Curtain|position|21|Percent|Study|Ground Floor"
            )[0],
            cover,
            cover_state,
        )
        self.assertEqual(integral.service_data["position"], 21)
        self.assertIsInstance(integral.service_data["position"], int)
        for raw in (
            "set|Curtain|position|20.9|Percent|Study|Ground Floor",
            "adjustUp|Curtain|position|10.5|Percent|Study|Ground Floor",
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(GroundingError, "integer"):
                build_plan(parse_domux_output(raw)[0], cover, cover_state)

        without_position = json.loads(json.dumps(cover_state))
        without_position["attributes"].pop("current_position")
        for raw, expected_state in (
            ("turnOn|Curtain|*|*|*|Study|Ground Floor", "open"),
            ("turnOff|Curtain|*|*|*|Study|Ground Floor", "closed"),
        ):
            with self.subTest(raw=raw):
                plan = build_plan(parse_domux_output(raw)[0], cover, without_position)
                self.assertEqual(plan.expected_projection, {
                    "entity_id": cover.entity_id,
                    "state": expected_state,
                })
        with self.assertRaisesRegex(GroundingError, "observed current_position"):
            build_plan(
                parse_domux_output(
                    "adjustUp|Curtain|position|10|Percent|Study|Ground Floor"
                )[0],
                cover,
                without_position,
            )

        light = self.registry.get("light.study_ceiling")
        light_state = json.loads(json.dumps(self.states[light.entity_id]))
        light_state["attributes"]["supported_color_modes"] = ["white"]
        for raw in (
            "set|Light|brightness|25|Percent|Study|Ground Floor",
            "adjustUp|Light|brightness|5|Percent|Study|Ground Floor",
        ):
            with self.subTest(raw=raw):
                plan = build_plan(parse_domux_output(raw)[0], light, light_state)
                self.assertEqual(plan.service, "turn_on")

        light_state["attributes"]["supported_color_modes"] = ["color_temp"]
        with self.assertRaisesRegex(GroundingError, "integer Kelvin"):
            build_plan(
                parse_domux_output(
                    "set|Light|colorTemperature|3000.9|Kelvin|Study|Ground Floor"
                )[0],
                light,
                light_state,
            )

    def test_climate_turn_off_prefers_advertised_off_mode_over_feature_gated_service(self) -> None:
        entity = self.registry.get("climate.study_ac")
        instruction = parse_domux_output(
            "turnOff|AC|*|*|*|Study|Ground Floor"
        )[0]
        state = json.loads(json.dumps(self.states[entity.entity_id]))
        self.assertEqual(state["attributes"]["supported_features"], 9)
        plan = build_plan(instruction, entity, state)
        self.assertEqual(plan.service, "set_hvac_mode")
        self.assertEqual(plan.service_data["hvac_mode"], "off")
        adapter = InMemoryHAAdapter({entity.entity_id: state})
        result = adapter.call_service(plan.domain, plan.service, plan.service_data)
        self.assertTrue(projection_matches(
            controlled_projection(result.after, "climate"),
            plan.expected_projection,
        ))

        state["attributes"]["hvac_modes"] = ["cool", "heat"]
        state["attributes"]["supported_features"] = 128
        feature_plan = build_plan(instruction, entity, state)
        self.assertEqual(feature_plan.service, "turn_off")
        self.assertNotIn("hvac_mode", feature_plan.service_data)

        state["attributes"]["supported_features"] = 0
        with self.assertRaisesRegex(GroundingError, "turn-off support"):
            build_plan(instruction, entity, state)

    def test_plan_rejects_noncanonical_units_and_unused_slots(self) -> None:
        adapter = InMemoryHAAdapter(self.states)
        entity = self.registry.get("light.study_ceiling")
        invalid = (
            "set|Light|brightness|30|Kelvin|Study|Ground Floor",
            "turnOff|Light|brightness|30|Percent|Study|Ground Floor",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(GroundingError):
                build_plan(parse_domux_output(raw)[0], entity, adapter.get_state(entity.entity_id))

    def test_climate_turn_on_requires_one_unambiguous_active_mode(self) -> None:
        adapter = InMemoryHAAdapter(self.states)
        off = adapter.get_state("climate.study_ac")
        off["state"] = "off"
        adapter.set_state_for_setup("climate.study_ac", off)
        instruction = parse_domux_output("turnOn|AC|*|*|*|Study|Ground Floor")[0]
        with self.assertRaisesRegex(GroundingError, "confirm a mode explicitly"):
            build_plan(instruction, self.registry.get("climate.study_ac"), off)
        off["attributes"]["hvac_modes"] = ["off", "heat"]
        plan = build_plan(instruction, self.registry.get("climate.study_ac"), off)
        self.assertEqual(plan.expected_projection["state"], "heat")
        receipt = adapter.call_service(plan.domain, plan.service, plan.service_data)
        self.assertEqual(receipt.after["state"], "heat")

    def test_climate_preserves_advertised_underscore_enums_and_temperature_step(self) -> None:
        state = json.loads(json.dumps(self.states["climate.study_ac"]))
        state["attributes"]["hvac_modes"] = ["off", "fan_only", "heat_cool"]
        entity = self.registry.get("climate.study_ac")
        fan_only = build_plan(
            parse_domux_output("set|AC|mode|Fan|*|Study|Ground Floor")[0], entity, state,
        )
        self.assertEqual(fan_only.service_data["hvac_mode"], "fan_only")
        fan_speed = build_plan(
            parse_domux_output("set|AC|fan speed|medium_high|Level|Study|Ground Floor")[0],
            entity,
            state,
        )
        self.assertEqual(fan_speed.service_data["fan_mode"], "medium_high")
        with self.assertRaisesRegex(GroundingError, "does not align"):
            build_plan(
                parse_domux_output("set|AC|temperature|23.3|Celsius|Study|Ground Floor")[0],
                entity,
                state,
            )
        accepted = build_plan(
            parse_domux_output("set|AC|temperature|23.5|Celsius|Study|Ground Floor")[0],
            entity,
            state,
        )
        self.assertEqual(accepted.service_data["temperature"], 23.5)
        with self.assertRaisesRegex(GroundingError, "does not align"):
            build_plan(
                parse_domux_output("adjustUp|AC|temperature|0.3|Celsius|Study|Ground Floor")[0],
                entity,
                state,
            )
        adjusted = build_plan(
            parse_domux_output("adjustUp|AC|temperature|0.5|Celsius|Study|Ground Floor")[0],
            entity,
            state,
        )
        self.assertEqual(adjusted.service_data["temperature"], 24.5)

    def test_projection_ignores_volatile_home_assistant_fields(self) -> None:
        raw = {
            "entity_id": "light.study_ceiling",
            "state": "on",
            "attributes": {"brightness": 153, "friendly_name": "private"},
            "last_changed": "volatile",
            "context": {"id": "volatile"},
        }
        self.assertEqual(
            controlled_projection(raw, "light"),
            {"entity_id": "light.study_ceiling", "state": "on", "brightness": 153},
        )
        planned = planning_projection(self.states["light.study_ceiling"], "light")
        self.assertIn("supported_color_modes", planned)

    def test_user_words_cannot_be_dropped_or_reversed_by_the_model(self) -> None:
        cases = (
            (
                "Turn off the Living Room light on the Ground Floor.",
                "turnOn|Light|*|*|*|Living Room|Ground Floor",
                "action",
            ),
            (
                "Open the Study curtain to 20 percent.",
                "turnOn|Curtain|*|*|*|Study|Ground Floor",
                "value",
            ),
            (
                "Set the Study AC to 23 Celsius.",
                "turnOn|AC|*|*|*|Study|Ground Floor",
                "action",
            ),
        )
        for utterance, raw, missing in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn(missing, grounded.clarification.unresolved_slots)

    def test_negated_target_and_cancelled_answers_fail_closed(self) -> None:
        grounded = ground_domux_request(
            "Turn off the Living Room light, not the Study light.",
            "turnOff|Light|*|*|*|Study|Ground Floor",
            self.registry,
        )
        confirmed = parse_domux_output(
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        )[0]
        with self.assertRaisesRegex(GroundingError, "excluded"):
            resolve_clarification_submission(
                grounded,
                answer="The Study light.",
                confirmed_instruction=confirmed,
                registry=self.registry,
            )

        uncertain = ground_domux_request(
            "Maybe turn off the Study light.",
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor",
            self.registry,
        )
        for answer in (
            "no thanks", "actually no", "cancel that", "never mind please", "not Study",
            "I do not know", "still not sure", "whatever", "maybe", "please ask me later", "banana",
        ):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    uncertain,
                    answer=answer,
                    confirmed_instruction=confirmed,
                    registry=self.registry,
                )

    def test_excluded_operation_values_require_a_positive_safe_replacement(self) -> None:
        cases = (
            (
                "Set Study light to anything but Blue.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
            (
                "Set Study light to something other than Blue.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
            (
                "Don't use Blue for Study light.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
            (
                "Set the Study AC to anything but Heat mode.",
                "set|AC|mode|Heat|*|Study|Ground Floor",
            ),
            (
                "Set brightness to any value other than 20% for the Study light.",
                "set|Light|brightness|20|Percent|Study|Ground Floor",
            ),
            (
            "Set color to avoid Blue for the Study light.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
            (
                "Make the Study light any color besides Blue.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
            (
                "Make the Study light any color apart from Blue.",
                "set|Light|color|Blue|*|Study|Ground Floor",
            ),
        )
        for utterance, raw in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertIn("excluded_operation_value", grounded.clarification.reasons)
                self.assertIn("value", grounded.clarification.unresolved_slots)
                for answer in ("Yes.", "The Study light."):
                    with self.assertRaisesRegex(GroundingError, "excluded"):
                        resolve_clarification_submission(
                            grounded,
                            answer=answer,
                            confirmed_instruction=parse_domux_output(raw)[0],
                            registry=self.registry,
                        )

        grounded = ground_domux_request(cases[0][0], cases[0][1], self.registry)
        safe = parse_domux_output(
            "set|Light|color|Red|*|Study|Ground Floor"
        )[0]
        resolved = resolve_clarification_submission(
            grounded,
            answer="Use Red instead for the Study light.",
            confirmed_instruction=safe,
            registry=self.registry,
        )
        self.assertEqual(resolved.confirmed_instruction.value, "Red")

    def test_withdrawn_initial_requests_and_deferred_answers_cannot_execute(self) -> None:
        raw = "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        for utterance in (
            "I don't want you to turn off the Study light.",
            "No need to turn off the Study light.",
            "Turn off the Study light, just kidding.",
            "Turn off the Study light—actually, never mind.",
            "Please refrain from turning off the Study light.",
            "Turn off the Study light, scratch that.",
            "Turn off the Study light, I changed my mind.",
            "Turn off the Study light, hold on.",
            "Dont turn off the Study light.",
            "Don’t turn off the Study light.",
            "Never ever turn off the Study light.",
        ):
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertIn("negative_or_cancelled_intent", grounded.clarification.reasons)
                with self.assertRaisesRegex(GroundingError, "negated"):
                    resolve_clarification_submission(
                        grounded,
                        answer="Yes.",
                        confirmed_instruction=parse_domux_output(raw)[0],
                        registry=self.registry,
                    )

        ambiguous = ground_domux_request(
            "Turn off the ceiling light.",
            "turnOff|Ceiling Light|*|*|*|*|*",
            self.registry,
        )
        confirmed = parse_domux_output(raw)[0]
        for answer in (
            "Study, leave it on.",
            "Study, keep it on.",
            "Study, not now.",
            "Study, wait.",
            "Study, please refrain from turning it off.",
            "The Study one, but don't do it yet.",
            "Study, forget it.",
            "Study, I changed my mind.",
            "Study, not anymore.",
            "Study, skip it.",
            "Study, don't.",
            "Study, do not.",
            "Study, abort it.",
            "Study, disregard that.",
            "Study, ignore that.",
            "Study, leave it alone.",
            "Study, no longer.",
            "Study, on second thought no.",
            "Study, postpone it.",
            "Study, defer it.",
            "Study, pause.",
            "Study, not yet.",
            "Study, don't touch it.",
            "Study, I don't want that.",
            "Study, no thanks.",
            "Study, scratch that.",
            "Study, nix that.",
            "Study, banana.",
            "Study, don't go ahead.",
            "Study, rather not proceed.",
            "Study, do not go ahead.",
            "Study, I do not want to proceed.",
            "Study, I don't want you to turn it off.",
            "Study, not turn it off.",
            "Study, I don't need you to turn it off.",
            "Study, I withdraw permission to turn it off.",
            "Study, I revoke authorization to turn it off.",
            "Study, I refuse to let you turn it off.",
            "Study, you must not turn it off.",
            "Study, you may not turn it off.",
            "Study, don't you turn it off.",
            "Study, I forbid you to turn it off.",
            "Study, under no circumstances turn it off.",
            "Use Study. Turn off the light if nobody is home.",
            "Use Study. Turn off the light at nine.",
            "Use Study. Turn off the light provided nobody is home.",
            "Use Study. Turn off the light, scratch that.",
            "Use Study, do not confirm.",
            "Use Study, don’t execute it.",
            "Study, should I turn it off?",
            "Study, can I turn it off?",
            "Study, would it be safe to turn it off?",
            "Study, is it okay to turn it off?",
            "Study, do you recommend I turn it off?",
            "Study, tell me whether to turn it off.",
            "Study, why should I turn it off?",
            "Study, are you going to turn it off?",
            "Use any device except Study. Turn it off.",
            "Not in Study; turn the light off.",
        ):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    ambiguous,
                    answer=answer,
                    confirmed_instruction=confirmed,
                    registry=self.registry,
                )
        for answer in ("Study.", "The one in the Study.", "Study, turn it off now."):
            with self.subTest(answer=answer):
                self.assertEqual(
                    resolve_clarification_submission(
                        ambiguous,
                        answer=answer,
                        confirmed_instruction=confirmed,
                        registry=self.registry,
                    ).chosen.entity_id,
                    "light.study_ceiling",
                )

    def test_clarification_answer_cannot_introduce_exclusions_or_alternatives(self) -> None:
        grounded = ground_domux_request(
            "Change the Study Ceiling Light color.",
            "set|Ceiling Light|color|*|*|Study|Ground Floor",
            self.registry,
        )
        blue = parse_domux_output(
            "set|Ceiling Light|color|Blue|*|Study|Ground Floor"
        )[0]
        for answer in (
            "Anything besides Blue.",
            "Anything apart from Blue.",
            "Use either Red or Blue.",
        ):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    grounded,
                    answer=answer,
                    confirmed_instruction=blue,
                    registry=self.registry,
                )

    def test_clarification_cannot_add_an_operation_that_the_plan_drops(self) -> None:
        grounded = ground_domux_request(
            "Turn off the Ceiling Light.",
            "turnOff|Ceiling Light|*|*|*|*|*",
            self.registry,
        )
        confirmed = parse_domux_output(
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        )[0]
        for answer in ("Study, make it Blue.", "Study, set brightness to 50 percent."):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    grounded,
                    answer=answer,
                    confirmed_instruction=confirmed,
                    registry=self.registry,
                )

        uncertain = ground_domux_request(
            "Maybe turn off the Study light.",
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor",
            self.registry,
        )
        with self.assertRaises(GroundingError):
            resolve_clarification_submission(
                uncertain,
                answer="Make it Blue.",
                confirmed_instruction=confirmed,
                registry=self.registry,
            )
        for answer in (
            "Study, maybe", "I do not know, Study?", "Study perhaps",
            "Study, but I am not sure", "whatever, Study",
        ):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    grounded,
                    answer=answer,
                    confirmed_instruction=confirmed,
                    registry=self.registry,
                )

    def test_named_room_is_not_misread_as_an_operational_color_or_mode(self) -> None:
        registry = EntityRegistry((
            EntitySpec("light.orange", "light", "Light", "Orange Room", "Ground Floor"),
            EntitySpec("climate.heat_room", "climate", "AC", "Heat Room", "Ground Floor"),
        ))
        grounded = ground_domux_request(
            "Turn off the Orange Room light.",
            "turnOff|Light|*|*|*|Orange Room|*",
            registry,
        )
        self.assertFalse(grounded.clarification.required)
        heat_room = ground_domux_request(
            "Turn off the Heat Room AC.",
            "turnOff|AC|*|*|*|Heat Room|*",
            registry,
        )
        self.assertFalse(heat_room.clarification.required)
        named_registry = EntityRegistry((
            EntitySpec("light.blue", "light", "Light", "Blue Room", "Ground Floor"),
            EntitySpec("light.study", "light", "Light", "Study", "Ground Floor"),
        ))
        named = ground_domux_request(
            "Turn off the light.", "turnOff|Light|*|*|*|*|*", named_registry,
        )
        resolved = resolve_clarification_submission(
            named,
            answer="Blue Room",
            confirmed_instruction=parse_domux_output(
                "turnOff|Light|*|*|*|Blue Room|Ground Floor"
            )[0],
            registry=named_registry,
        )
        self.assertEqual(resolved.chosen.entity_id, "light.blue")

    def test_opposing_action_clauses_cannot_collapse_to_one_model_action(self) -> None:
        for utterance in (
            "Turn on the Study light then switch that device off.",
            "Turn on the Study light, then turn the Study light off.",
        ):
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(
                    utterance,
                    "turnOn|Ceiling Light|*|*|*|Study|Ground Floor",
                    self.registry,
                )
                self.assertTrue(grounded.clarification.required)
                self.assertIn("action", grounded.clarification.unresolved_slots)

    def test_source_values_ranges_and_multi_operation_text_fail_closed(self) -> None:
        cases = (
            (
                "Change the Study light brightness from 50 to 20 percent.",
                "set|Light|brightness|50|Percent|Study|*",
                "value",
            ),
            (
                "Set the Study light brightness between 20 and 50 percent.",
                "set|Light|brightness|50|Percent|Study|*",
                "value",
            ),
            (
                "Change the Study light color from Red to Blue.",
                "set|Light|color|Red|*|Study|*",
                "value",
            ),
            (
                "Use Heat then Cool mode on the Study AC.",
                "set|AC|mode|Heat|*|Study|*",
                "value",
            ),
            (
                "Open the Study curtain halfway, then close it.",
                "set|Curtain|position|50|Percent|Study|*",
                "action",
            ),
            (
                "Close the Study curtain, then open it to 20 percent.",
                "set|Curtain|position|20|Percent|Study|*",
                "action",
            ),
            (
                "Set the Study AC to Cool mode at 20.",
                "set|AC|mode|Cool|*|Study|*",
                "value",
            ),
        )
        for utterance, raw, slot in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn(slot, grounded.clarification.unresolved_slots)

        target = ground_domux_request(
            "Change the Study light brightness from 50 to 20 percent.",
            "set|Light|brightness|20|Percent|Study|*",
            self.registry,
        )
        color_target = ground_domux_request(
            "Change the Study light color from Red to Blue.",
            "set|Light|color|Blue|*|Study|*",
            self.registry,
        )
        self.assertFalse(target.clarification.required)
        self.assertFalse(color_target.clarification.required)

    def test_explicit_temperature_unit_cannot_be_silently_changed(self) -> None:
        for unit in ("Fahrenheit", "°F", "Kelvin", "K"):
            with self.subTest(unit=unit):
                grounded = ground_domux_request(
                    f"Set the Study AC temperature to 20 {unit}.",
                    "set|AC|temperature|20|Celsius|Study|*",
                    self.registry,
                )
                self.assertTrue(grounded.clarification.required)
                self.assertIn("unit", grounded.clarification.unresolved_slots)

    def test_conflicted_action_and_unit_can_converge_after_explicit_clarification(self) -> None:
        action_grounded = ground_domux_request(
            "Open the Study curtain halfway, then close it.",
            "set|Curtain|position|50|Percent|Study|*",
            self.registry,
        )
        action_resolved = resolve_clarification_submission(
            action_grounded,
            answer="Close the Study curtain.",
            confirmed_instruction=parse_domux_output(
                "turnOff|Curtain|*|*|*|Study|Ground Floor"
            )[0],
            registry=self.registry,
        )
        self.assertEqual(action_resolved.confirmed_instruction.action, "turnOff")

        unit_grounded = ground_domux_request(
            "Set the Study AC temperature to 20 degrees Fahrenheit.",
            "set|AC|temperature|20|Celsius|Study|*",
            self.registry,
        )
        unit_resolved = resolve_clarification_submission(
            unit_grounded,
            answer="Use 20 Celsius for the Study AC instead.",
            confirmed_instruction=parse_domux_output(
                "set|AC|temperature|20|Celsius|Study|Ground Floor"
            )[0],
            registry=self.registry,
        )
        self.assertEqual(unit_resolved.confirmed_instruction.unit, "Celsius")

    def test_compound_effects_and_multiple_attributes_cannot_collapse(self) -> None:
        cases = (
            (
                "Turn off the Study light and make it Blue.",
                "set|Light|color|Blue|*|Study|*",
                "action",
            ),
            (
                "Turn off the Study light and set brightness to 20 percent.",
                "set|Light|brightness|20|Percent|Study|*",
                "action",
            ),
            (
                "Close the Study curtain and set position to 20 percent.",
                "set|Curtain|position|20|Percent|Study|*",
                "action",
            ),
            (
                "Turn off the Study AC and use Cool mode.",
                "set|AC|mode|Cool|*|Study|*",
                "action",
            ),
            (
                "Turn off the Study light and make it brighter.",
                "adjustUp|Light|brightness|10|Percent|Study|*",
                "action",
            ),
            (
                "Close the Study curtain and raise it.",
                "adjustUp|Curtain|position|10|Percent|Study|*",
                "action",
            ),
            (
                "Set the Study light brightness and color to Blue.",
                "set|Light|color|Blue|*|Study|*",
                "attribute",
            ),
            (
                "Make the Study AC warmer and use Cool mode.",
                "set|AC|mode|Cool|*|Study|*",
                "attribute",
            ),
            (
                "Set the Study AC wind speed High and use Cool mode.",
                "set|AC|mode|Cool|*|Study|*",
                "attribute",
            ),
        )
        for utterance, raw, slot in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn(slot, grounded.clarification.unresolved_slots)

        compatible = ground_domux_request(
            "Turn on the Study light and make it Blue.",
            "set|Light|color|Blue|*|Study|*",
            self.registry,
        )
        self.assertFalse(compatible.clarification.required)

    def test_explicit_multi_targets_cannot_be_truncated_to_one_tuple(self) -> None:
        cases = (
            (
                "Turn off the Living Room light and the Study light.",
                "turnOff|Light|*|*|*|Living Room|*",
                {"light.living_ceiling", "light.study_ceiling"},
            ),
            (
                "Turn off the Study light and AC.",
                "turnOff|Light|*|*|*|Study|*",
                {"light.study_ceiling", "climate.study_ac"},
            ),
            (
                "Turn on the Study light and open the curtain.",
                "turnOn|Light|*|*|*|Study|*",
                {"light.study_ceiling", "cover.study_curtain"},
            ),
        )
        for utterance, raw, expected in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertTrue(expected.issubset({item.entity_id for item in grounded.candidates}))

    def test_nested_room_names_select_only_the_longest_explicit_span(self) -> None:
        nested = EntityRegistry((
            EntitySpec("light.guest", "light", "Light", "Guest Bedroom", "Ground Floor"),
            EntitySpec("light.bedroom", "light", "Light", "Bedroom", "Ground Floor"),
            EntitySpec("light.east_hall", "light", "Light", "East Hall", "First Floor"),
            EntitySpec("light.hall", "light", "Light", "Hall", "First Floor"),
        ))
        for room, entity_id in (
            ("Guest Bedroom", "light.guest"),
            ("East Hall", "light.east_hall"),
        ):
            with self.subTest(room=room):
                grounded = ground_domux_request(
                    f"Turn off the {room} light.",
                    f"turnOff|Light|*|*|*|{room}|*",
                    nested,
                )
                self.assertFalse(grounded.clarification.required)
                self.assertEqual(grounded.candidates[0].entity_id, entity_id)

        both = ground_domux_request(
            "Turn off the Guest Bedroom light and the Bedroom light.",
            "turnOff|Light|*|*|*|Guest Bedroom|*",
            nested,
        )
        self.assertTrue(both.clarification.required)
        self.assertTrue({"light.guest", "light.bedroom"}.issubset(
            {entity.entity_id for entity in both.candidates}
        ))

    def test_numeric_specific_device_labels_are_not_operation_values(self) -> None:
        numeric = EntityRegistry((
            EntitySpec("light.lamp_2", "light", "Lamp 2", "Study", "Ground Floor"),
            EntitySpec("cover.curtain_2", "cover", "Curtain 2", "Study", "Ground Floor"),
            EntitySpec("climate.ac_2", "climate", "AC 2", "Study", "Ground Floor"),
        ))
        for utterance, raw in (
            ("Turn off the Study Lamp 2.", "turnOff|Lamp 2|*|*|*|Study|*"),
            ("Close the Study Curtain 2.", "turnOff|Curtain 2|*|*|*|Study|*"),
            ("Turn off the Study AC 2.", "turnOff|AC 2|*|*|*|Study|*"),
        ):
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, numeric)
                self.assertFalse(grounded.clarification.required)

    def test_explicit_alternatives_require_value_clarification(self) -> None:
        cases = (
            (
                "Set the Study light brightness to 20 or 50 percent.",
                "set|Light|brightness|20|Percent|Study|*",
            ),
            (
                "Set the Study light color to Blue or Red.",
                "set|Light|color|Blue|*|Study|*",
            ),
            (
                "Set the Study AC mode to Heat or Cool.",
                "set|AC|mode|Heat|*|Study|*",
            ),
            (
                "Set the Study light brightness to 20 and 50 percent.",
                "set|Light|brightness|20|Percent|Study|*",
            ),
            (
                "Set the Study light brightness to 20, 50 percent.",
                "set|Light|brightness|20|Percent|Study|*",
            ),
            (
                "Set the Study light color to Blue with a Red accent.",
                "set|Light|color|Blue|*|Study|*",
            ),
            (
                "Set the Study AC mode to Cool with Fan Only.",
                "set|AC|mode|Cool|*|Study|*",
            ),
            (
                "Set the Study light brightness below 20 percent.",
                "set|Light|brightness|20|Percent|Study|*",
            ),
            (
                "Set the Study AC temperature above 20 Celsius.",
                "set|AC|temperature|20|Celsius|Study|*",
            ),
        )
        for utterance, raw in cases:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn("value", grounded.clarification.unresolved_slots)

    def test_fan_speed_is_not_misread_as_an_hvac_mode(self) -> None:
        for attribute in ("fan speed", "wind speed"):
            with self.subTest(attribute=attribute):
                grounded = ground_domux_request(
                    f"Set the Study AC {attribute} to High.",
                    f"set|AC|{attribute}|High|Level|Study|*",
                    self.registry,
                )
                self.assertFalse(grounded.clarification.required)

    def test_absolute_and_relative_numeric_actions_cannot_be_interchanged(self) -> None:
        rejected = (
            ("Raise the Study light brightness to 20 percent.", "adjustUp|Light|brightness|20|Percent|Study|*"),
            ("Lower the Study light brightness to 20 percent.", "adjustDown|Light|brightness|20|Percent|Study|*"),
            ("Make the Study AC warmer to 25 Celsius.", "adjustUp|AC|temperature|25|Celsius|Study|*"),
            ("Raise the Study curtain to 30 percent.", "adjustUp|Curtain|position|30|Percent|Study|*"),
            ("Set the Study light brightness by 20 percent.", "set|Light|brightness|20|Percent|Study|*"),
            ("Open the Study curtain by 20 percent.", "set|Curtain|position|20|Percent|Study|*"),
        )
        for utterance, raw in rejected:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn("value", grounded.clarification.unresolved_slots)

        accepted = (
            ("Increase the Study light brightness by 20 percent.", "adjustUp|Light|brightness|20|Percent|Study|*"),
            ("Make the Study AC 2 degrees warmer.", "adjustUp|AC|temperature|2|Celsius|Study|*"),
        )
        for utterance, raw in accepted:
            with self.subTest(utterance=utterance):
                self.assertFalse(ground_domux_request(utterance, raw, self.registry).clarification.required)

    def test_climate_turn_on_requires_a_resolvable_mode_clarification(self) -> None:
        grounded = ground_domux_request(
            "Turn on the Study AC.",
            "turnOn|AC|*|*|*|Study|*",
            self.registry,
        )
        self.assertTrue(grounded.clarification.required)
        self.assertIn("climate_mode_confirmation_required", grounded.clarification.reasons)
        resolved = resolve_clarification_submission(
            grounded,
            answer="Use Cool mode on the Study AC.",
            confirmed_instruction=parse_domux_output(
                "set|AC|mode|Cool|*|Study|Ground Floor"
            )[0],
            registry=self.registry,
        )
        plan = build_plan(
            resolved.confirmed_instruction,
            resolved.chosen,
            self.states[resolved.chosen.entity_id],
        )
        self.assertEqual((plan.service, plan.service_data["hvac_mode"]), ("set_hvac_mode", "cool"))

    def test_compound_turn_on_must_match_the_service_side_effect(self) -> None:
        rejected = (
            (
                "Open the Study curtain to 0 percent.",
                "set|Curtain|position|0|Percent|Study|*",
            ),
            (
                "Turn on the Study AC and set temperature to 20 Celsius.",
                "set|AC|temperature|20|Celsius|Study|*",
            ),
            (
                "Turn on the Study AC and set fan speed to High.",
                "set|AC|fan speed|High|Level|Study|*",
            ),
        )
        for utterance, raw in rejected:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertTrue(grounded.clarification.required)
                self.assertIn("action", grounded.clarification.unresolved_slots)

        accepted = (
            (
                "Open the Study curtain to 20 percent.",
                "set|Curtain|position|20|Percent|Study|*",
            ),
            (
                "Turn on the Study AC and use Cool mode.",
                "set|AC|mode|Cool|*|Study|*",
            ),
        )
        for utterance, raw in accepted:
            with self.subTest(utterance=utterance):
                self.assertFalse(ground_domux_request(utterance, raw, self.registry).clarification.required)

    def test_conditions_schedules_durations_and_meta_questions_fail_closed(self) -> None:
        conditionals = (
            "Turn off the Study light if the room is empty.",
            "Turn on the Study light when it gets dark.",
            "Turn off the Study light after dinner.",
            "Turn off the Study light tomorrow.",
            "Turn on the Study light for an hour.",
            "Turn off the Study light at noon.",
            "Turn off the Study light in five minutes.",
            "Turn off the Study light once I leave.",
            "Turn off the Study light every night.",
            "Turn off the Study light at 6:30.",
            "Turn off the Study light provided nobody is home.",
            "Turn off the Study light as long as nobody is home.",
            "Turn off the Study light in half an hour.",
            "Turn off the Study light on Monday.",
            "Turn off the Study light at nine.",
        )
        for utterance in conditionals:
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(
                    utterance, "turnOff|Light|*|*|*|Study|*", self.registry,
                )
                self.assertTrue(grounded.clarification.required)
                self.assertIn("unsupported_condition_or_time", grounded.clarification.reasons)
                with self.assertRaisesRegex(GroundingError, "immediate command"):
                    resolve_clarification_submission(
                        grounded,
                        answer="Yes, the Study light.",
                        confirmed_instruction=parse_domux_output(
                            "turnOff|Light|*|*|*|Study|Ground Floor"
                        )[0],
                        registry=self.registry,
                    )

        for utterance in (
            "Should I turn off the Study light?",
            "Tell me how to turn off the Study light.",
            "What happens if I turn off the Study light?",
            "Can I turn off the Study light?",
            "Is it okay to turn off the Study light?",
            "Do you recommend I turn off the Study light?",
            "Why should I turn off the Study light?",
            "Please explain why I should turn off the Study light?",
            "Do I need to turn off the Study light?",
            "Would it be safe to turn off the Study light?",
        ):
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(
                    utterance, "turnOff|Light|*|*|*|Study|*", self.registry,
                )
                self.assertIn("informational_request", grounded.clarification.reasons)
                with self.assertRaisesRegex(GroundingError, "informational"):
                    resolve_clarification_submission(
                        grounded,
                        answer="Yes, do it now.",
                        confirmed_instruction=parse_domux_output(
                            "turnOff|Light|*|*|*|Study|Ground Floor"
                        )[0],
                        registry=self.registry,
                    )

        for utterance in (
            "Can you turn off the Study light?",
            "Could you turn off the Study light?",
        ):
            with self.subTest(utterance=utterance):
                polite = ground_domux_request(
                    utterance,
                    "turnOff|Light|*|*|*|Study|*",
                    self.registry,
                )
                self.assertFalse(polite.clarification.required)

    def test_unconsumed_initial_language_cannot_be_recovered_by_clarification(self) -> None:
        modifiers = (
            "subject to nobody being home",
            "depending on whether anyone is home",
            "so long as the room is empty",
            "during dinner",
            "upon my arrival",
            "at dusk",
            "at dawn",
            "this Friday",
            "next Friday",
            "by nine",
            "for the next hour",
            "momentarily",
            "as needed",
            "forget it",
            "I withdraw that",
            "I revoke that",
            "I take that back",
            "don't bother",
            "actually don't",
            "use all but Blue",
            "use Blue excluding Green",
            "use non-Blue",
        )
        confirmed = parse_domux_output(
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        )[0]
        for modifier in modifiers:
            utterance = f"Turn off the Study Ceiling Light, {modifier}."
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(
                    utterance,
                    "turnOff|Ceiling Light|*|*|*|Study|Ground Floor",
                    self.registry,
                )
                self.assertTrue(grounded.clarification.required)
                with self.assertRaises(GroundingError):
                    resolve_clarification_submission(
                        grounded,
                        answer="The Study Ceiling Light.",
                        confirmed_instruction=confirmed,
                        registry=self.registry,
                    )

        for utterance in (
            "Turn off the Study Ceiling Light; wait.",
            "Turn off the Study Ceiling Light, stop.",
            "Turn off the Study Ceiling Light, do not.",
            "Turn off the Study Ceiling Light, no.",
            "Turn off the Study Ceiling Light, not now.",
            "Turn off the Study Ceiling Light; I don't want it.",
            "Turn off the Study Ceiling Light; I dont want it.",
            "Turn off the Study Ceiling Light; I don't need it.",
            "Turn off the Study Ceiling Light; I dont need it.",
            "Turn off the Study Ceiling Light; please do not.",
            "Turn off the Study Ceiling Light; please no.",
            "Turn off the Study Ceiling Light; no please.",
            "Turn off the Study Ceiling Light; I mean no.",
            "Turn off the Study Ceiling Light; I want no.",
        ):
            with self.subTest(utterance=utterance):
                grounded = ground_domux_request(
                    utterance,
                    "turnOff|Ceiling Light|*|*|*|Study|Ground Floor",
                    self.registry,
                )
                self.assertIn("negative_or_cancelled_intent", grounded.clarification.reasons)

        for utterance in (
            "Turn off the Study Ceiling Light, use the other one.",
            "Turn off the Study Ceiling Light, use the other light.",
            "Turn off the Study Ceiling Light, not this one.",
            "Turn off the Study Ceiling Light, not that one.",
        ):
            with self.subTest(utterance=utterance):
                generic = ground_domux_request(
                    utterance,
                    "turnOff|Ceiling Light|*|*|*|Study|Ground Floor",
                    self.registry,
                )
                self.assertIn("unsupported_request_grammar", generic.clarification.reasons)

    def test_model_slots_cannot_launder_unconsumed_request_language(self) -> None:
        utterance = (
            "Set the Study Ceiling Light brightness to 50 percent "
            "subject to nobody being home."
        )
        raws = (
            "set|Ceiling Light|brightness|50|subject to nobody being home|Study|*",
            "set|Ceiling Light|subject to nobody being home|50|Percent|Study|*",
        )
        confirmed = parse_domux_output(
            "set|Ceiling Light|brightness|50|Percent|Study|Ground Floor"
        )[0]
        for raw in raws:
            with self.subTest(raw=raw):
                grounded = ground_domux_request(utterance, raw, self.registry)
                self.assertIn("unsupported_request_grammar", grounded.clarification.reasons)
                with self.assertRaisesRegex(GroundingError, "new immediate command"):
                    resolve_clarification_submission(
                        grounded,
                        answer="Study, confirm 50 percent brightness.",
                        confirmed_instruction=confirmed,
                        registry=self.registry,
                    )

    def test_displayed_candidate_indices_are_selectors_not_operation_values(self) -> None:
        registry = EntityRegistry((
            EntitySpec("light.alpha", "light", "Light", "Alpha", "Ground Floor"),
            EntitySpec("light.beta", "light", "Light", "Beta", "Ground Floor"),
            EntitySpec("light.gamma", "light", "Light", "Gamma", "Ground Floor"),
        ))
        grounded = ground_domux_request(
            "Turn off the light.",
            "turnOff|Light|*|*|*|*|*",
            registry,
        )
        self.assertEqual(len(grounded.clarification.candidates), 3)
        self.assertFalse(
            {"action", "attribute", "value", "unit"}
            .intersection(grounded.clarification.unresolved_slots)
        )
        for index, chosen in enumerate(grounded.candidates, start=1):
            with self.subTest(index=index, entity_id=chosen.entity_id):
                confirmed = DomuxInstruction(
                    "turnOff", "Light", "*", "*", "*", chosen.room, chosen.floor,
                )
                resolved = resolve_clarification_submission(
                    grounded,
                    answer=str(index),
                    confirmed_instruction=confirmed,
                    registry=registry,
                )
                self.assertEqual(resolved.chosen.entity_id, chosen.entity_id)

    def test_generic_selector_reversals_fail_closed(self) -> None:
        grounded = ground_domux_request(
            "Turn off the ceiling light.",
            "turnOff|Ceiling Light|*|*|*|*|*",
            self.registry,
        )
        confirmed = parse_domux_output(
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        )[0]
        for answer in (
            "Study, not this one.",
            "Study, not that one.",
            "Study, not this device.",
            "Study, not the one I mean.",
            "The other one.",
            "Use the other one.",
            "Leave this one unchanged.",
            "Study, not this.",
            "I mean the other one.",
        ):
            with self.subTest(answer=answer), self.assertRaises(GroundingError):
                resolve_clarification_submission(
                    grounded,
                    answer=answer,
                    confirmed_instruction=confirmed,
                    registry=self.registry,
                )

    def test_candidate_selector_text_cannot_double_as_operation_authorization(self) -> None:
        grounded = ground_domux_request(
            "Set the light brightness between 1 and 20 percent.",
            "set|Light|brightness|1|Percent|*|*",
            self.registry,
        )
        with self.assertRaises(GroundingError):
            resolve_clarification_submission(
                grounded,
                answer="2",
                confirmed_instruction=parse_domux_output(
                    "set|Light|brightness|2|Percent|Study|Ground Floor"
                )[0],
                registry=self.registry,
            )

        numbered = EntityRegistry((
            EntitySpec("light.bedroom_50", "light", "Light", "Bedroom", "Ground Floor"),
            EntitySpec("light.study_numbered", "light", "Light", "Study", "Ground Floor"),
        ))
        numbered_grounded = ground_domux_request(
            "Set the light brightness between 20 and 80 percent.",
            "set|Light|brightness|20|Percent|*|*",
            numbered,
        )
        with self.assertRaises(GroundingError):
            resolve_clarification_submission(
                numbered_grounded,
                answer="light.bedroom_50",
                confirmed_instruction=parse_domux_output(
                    "set|Light|brightness|50|Percent|Bedroom|Ground Floor"
                )[0],
                registry=numbered,
            )

    def test_selector_words_and_numbers_cannot_masquerade_as_operation_values(self) -> None:
        numbered = EntityRegistry((
            EntitySpec("light.bedroom_50", "light", "Light", "Bedroom 50", "Ground Floor"),
        ))
        utterance = "Set the Bedroom 50 light brightness to 20 percent."
        wrong = ground_domux_request(
            utterance, "set|Light|brightness|50|Percent|Bedroom 50|*", numbered,
        )
        right = ground_domux_request(
            utterance, "set|Light|brightness|20|Percent|Bedroom 50|*", numbered,
        )
        self.assertTrue(wrong.clarification.required)
        self.assertIn("value", wrong.clarification.unresolved_slots)
        self.assertFalse(right.clarification.required)

        collision_registry = EntityRegistry((
            EntitySpec("light.orange_room", "light", "Light", "Orange Room", "Ground Floor"),
            EntitySpec("climate.heat_room", "climate", "AC", "Heat Room", "Ground Floor"),
        ))
        color_wrong = ground_domux_request(
            "Make the Orange Room light brighter.",
            "set|Light|color|Orange|*|Orange Room|*",
            collision_registry,
        )
        mode_wrong = ground_domux_request(
            "Make the Heat Room AC warmer.",
            "set|AC|mode|Heat|*|Heat Room|*",
            collision_registry,
        )
        self.assertTrue(color_wrong.clarification.required)
        self.assertIn("attribute", color_wrong.clarification.unresolved_slots)
        self.assertTrue(mode_wrong.clarification.required)
        self.assertIn("attribute", mode_wrong.clarification.unresolved_slots)


class FailOnceReadAdapter(InMemoryHAAdapter):
    fail_next_read = False

    def get_state(self, entity_id: str) -> dict[str, object]:
        if self.fail_next_read:
            self.fail_next_read = False
            raise AdapterError("injected predispatch read failure")
        return super().get_state(entity_id)


class ClockAdvancingAdapter(InMemoryHAAdapter):
    def __init__(self, states: dict[str, dict[str, object]], clock: MutableClock):
        super().__init__(states)
        self.clock = clock
        self.advance_reads = False

    def get_state(self, entity_id: str) -> dict[str, object]:
        value = super().get_state(entity_id)
        if self.advance_reads:
            self.clock.value += 8
        return value


class UnknownOutcomeAdapter(InMemoryHAAdapter):
    def call_service(self, domain: str, service: str, data: dict[str, object]) -> ServiceCallResult:
        self.sut_calls.append({"kind": "sut", "outcome": "request_error_outcome_unknown"})
        raise ServiceCallError(
            "injected transport loss", attempted=True, acknowledged=False, outcome_unknown=True,
        )


class WaitFailureAdapter(InMemoryHAAdapter):
    def wait_for_projection(
        self, entity_id: str, domain: str, expected: dict[str, object],
    ) -> dict[str, object]:
        del entity_id, domain, expected
        raise AdapterError("injected post-ack observation failure")


class BlockingAdapter(InMemoryHAAdapter):
    def __init__(self, states: dict[str, dict[str, object]]):
        super().__init__(states)
        self.utility_started = threading.Event()
        self.release_utility = threading.Event()

    def call_service(self, domain: str, service: str, data: dict[str, object]) -> ServiceCallResult:
        if data.get("entity_id") == "light.utility":
            self.utility_started.set()
            if not self.release_utility.wait(timeout=2):
                raise AssertionError("test did not release the blocked utility call")
        return super().call_service(domain, service, data)


class PreparedActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry, self.states = fixture()
        self.grounded = ground_domux_request(
            "Turn off the Ceiling Light.",
            "turnOff|Ceiling Light|*|*|*|*|*",
            self.registry,
        )
        self.confirmed = parse_domux_output(
            "turnOff|Ceiling Light|*|*|*|Study|Ground Floor"
        )[0]
        self.clock = MutableClock()
        self.nonce_index = 0

    def nonce(self) -> str:
        self.nonce_index += 1
        return f"test-nonce-{self.nonce_index}"

    def prepare(
        self,
        store_type: type[PreparedActionStore] = PreparedActionStore,
        *,
        adapter: InMemoryHAAdapter | None = None,
        grounded=None,
        answer: str | None = "Study",
        confirmed: DomuxInstruction | None = None,
        state_dependencies: tuple[str, ...] = (),
    ):
        adapter = adapter or InMemoryHAAdapter(self.states)
        store = store_type(ttl_seconds=30, clock=self.clock, nonce_factory=self.nonce)
        grounded = grounded or self.grounded
        action = store.prepare(
            actor_id="actor-a",
            session_id="session-a",
            grounded=grounded,
            registry=self.registry,
            adapter=adapter,
            clarification_answer=answer,
            confirmed_instruction=confirmed or self.confirmed,
            state_dependencies=state_dependencies,
        )
        return adapter, store, action

    def test_clean_commit_changes_only_the_selected_entity(self) -> None:
        adapter, store, action = self.prepare()
        untouched = adapter.get_state("light.living_ceiling")
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertTrue(result.accepted and result.dispatched and result.acknowledged)
        self.assertEqual((result.status, result.after["state"]), ("COMMITTED", "off"))
        self.assertEqual(adapter.get_state("light.living_ceiling"), untouched)
        self.assertEqual(len(adapter.sut_calls), 1)

    def test_public_handle_confirmation_and_plan_copies_are_immutable(self) -> None:
        adapter, store, action = self.prepare()
        with self.assertRaises(FrozenInstanceError):
            action.entity_id = "light.living_ceiling"  # type: ignore[misc]
        snapshot = store.snapshot(action.nonce)
        snapshot["plan"]["service_data"]["entity_id"] = "light.living_ceiling"
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((result.status, result.after["entity_id"]), ("COMMITTED", "light.study_ceiling"))
        tombstone = store.snapshot(action.nonce)
        self.assertTrue(tombstone["redacted"])
        retained = json.dumps(tombstone)
        self.assertNotIn("Turn off the Ceiling Light", retained)
        self.assertNotIn("light.study_ceiling", retained)
        self.assertNotIn("actor-a", retained)

    def test_replay_and_two_prepared_nonces_dispatch_at_most_once_each_state(self) -> None:
        adapter, store, first = self.prepare()
        second = store.prepare(
            actor_id="actor-a", session_id="session-a", grounded=self.grounded,
            registry=self.registry, adapter=adapter, clarification_answer="Study",
            confirmed_instruction=self.confirmed,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda item: store.commit(item.confirmation(), registry=self.registry, adapter=adapter),
                (first, second),
            ))
        self.assertEqual(sum(result.dispatched for result in results), 1)
        self.assertEqual({result.reason for result in results}, {"committed", "state_changed"})
        replay = store.commit(first.confirmation(), registry=self.registry, adapter=adapter)
        self.assertIn(replay.reason, {"replayed_nonce", "action_not_prepared"})
        self.assertEqual(len(adapter.sut_calls), 1)

    def test_expiry_state_capability_and_candidate_drift_are_zero_dispatch(self) -> None:
        adapter, store, action = self.prepare()
        self.clock.value = action.expires_at + 1
        self.assertEqual(
            store.commit(action.confirmation(), registry=self.registry, adapter=adapter).reason,
            "expired",
        )

        self.clock.value = 1000
        adapter, store, action = self.prepare()
        adapter.mutate_state_for_setup("light.study_ceiling")
        self.assertEqual(store.commit(action.confirmation(), registry=self.registry, adapter=adapter).reason, "state_changed")

        adapter, store, action = self.prepare()
        changed_state = adapter.get_state("light.study_ceiling")
        changed_state["attributes"]["supported_color_modes"] = ["onoff"]
        adapter.set_state_for_setup("light.study_ceiling", changed_state)
        self.assertEqual(store.commit(action.confirmation(), registry=self.registry, adapter=adapter).reason, "state_changed")

        adapter, store, action = self.prepare()
        changed = self.registry.with_replacement(
            EntitySpec("light.study_ceiling", "light", "Ceiling Light", "Library", "Ground Floor")
        )
        self.assertEqual(store.commit(action.confirmation(), registry=changed, adapter=adapter).reason, "candidate_set_changed")

        adapter, store, action = self.prepare()
        expanded = EntityRegistry((*self.registry.entities, EntitySpec(
            "light.bedroom_ceiling", "light", "Ceiling Light", "Bedroom", "Ground Floor",
        )))
        self.assertEqual(store.commit(action.confirmation(), registry=expanded, adapter=adapter).reason, "candidate_set_changed")
        self.assertEqual(len(adapter.sut_calls), 0)

    def test_confirmation_binds_all_authorization_digests(self) -> None:
        mutations = (
            ("actor_id", "actor-b", "actor_mismatch"),
            ("session_id", "session-b", "session_mismatch"),
            ("request_digest", "0" * 64, "request_mismatch"),
            ("clarification_digest", "0" * 64, "clarification_mismatch"),
            ("plan_digest", "0" * 64, "plan_mismatch"),
            ("candidate_digest", "0" * 64, "confirmation_candidate_mismatch"),
        )
        for field, value, reason in mutations:
            adapter, store, action = self.prepare()
            result = store.commit(
                altered_confirmation(action.confirmation(), **{field: value}),
                registry=self.registry,
                adapter=adapter,
            )
            self.assertEqual((result.reason, len(adapter.sut_calls)), (reason, 0))

    def test_only_declared_or_context_state_is_bound(self) -> None:
        adapter, store, action = self.prepare()
        adapter.mutate_state_for_setup("light.utility")
        self.assertTrue(store.commit(action.confirmation(), registry=self.registry, adapter=adapter).accepted)

        context_grounded = ground_domux_request(
            "Turn off that device.",
            "turnOff|*|*|*|*|*|*",
            self.registry,
            SessionContext(("light.study_ceiling", "light.utility")),
        )
        adapter, store, action = self.prepare(grounded=context_grounded)
        adapter.mutate_state_for_setup("light.utility")
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((result.reason, len(adapter.sut_calls)), ("state_changed", 0))

        adapter, store, action = self.prepare(state_dependencies=("light.utility",))
        adapter.mutate_state_for_setup("light.utility")
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((result.reason, len(adapter.sut_calls)), ("state_changed", 0))

    def test_slow_predispatch_reads_cannot_cross_ttl(self) -> None:
        adapter = ClockAdvancingAdapter(self.states, self.clock)
        adapter, store, action = self.prepare(adapter=adapter)
        adapter.advance_reads = True
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((result.reason, len(adapter.sut_calls)), ("expired", 0))

    def test_abandoned_expired_action_is_redacted_on_purge(self) -> None:
        _adapter, store, action = self.prepare()
        self.clock.value = action.expires_at + 1
        self.assertEqual(store.purge_expired(), 1)
        snapshot = store.snapshot(action.nonce)
        self.assertTrue(snapshot["redacted"])
        retained = json.dumps(snapshot)
        self.assertNotIn("Ceiling Light", retained)
        self.assertNotIn("light.study_ceiling", retained)

    def test_predispatch_read_failure_does_not_consume_nonce(self) -> None:
        adapter = FailOnceReadAdapter(self.states)
        adapter, store, action = self.prepare(adapter=adapter)
        adapter.fail_next_read = True
        first = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((first.reason, first.dispatched), ("predispatch_state_read_failed", False))
        second = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((second.status, len(adapter.sut_calls)), ("COMMITTED", 1))

    def test_dispatch_and_post_ack_unknown_outcomes_are_action_local(self) -> None:
        adapter, store, action = self.prepare(adapter=UnknownOutcomeAdapter(self.states))
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual(
            (result.status, result.dispatched, result.acknowledged, result.outcome_unknown),
            ("FAILED_DISPATCH", True, False, True),
        )

        adapter, store, action = self.prepare(adapter=WaitFailureAdapter(self.states))
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual(
            (result.status, result.dispatched, result.acknowledged, result.outcome_unknown),
            ("FAILED_POSTCONDITION", True, True, True),
        )

    def test_postcondition_failure_is_visible_and_nonce_stays_consumed(self) -> None:
        adapter, store, action = self.prepare()
        adapter.force_postcondition_mismatch = True
        result = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((result.status, result.reason), ("FAILED_POSTCONDITION", "postcondition_mismatch"))
        replay = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertEqual((replay.reason, len(adapter.sut_calls)), ("replayed_nonce", 1))

    def test_dependency_target_race_is_serialized_before_dispatch(self) -> None:
        adapter = BlockingAdapter(self.states)
        store = PreparedActionStore(ttl_seconds=30, clock=self.clock, nonce_factory=self.nonce)
        context_grounded = ground_domux_request(
            "Turn off that device.", "turnOff|*|*|*|*|*|*", self.registry,
            SessionContext(("light.study_ceiling", "light.utility")),
        )
        dependent = store.prepare(
            actor_id="actor-a", session_id="session-a", grounded=context_grounded,
            registry=self.registry, adapter=adapter, clarification_answer="Study",
            confirmed_instruction=self.confirmed,
        )
        utility_grounded = ground_domux_request(
            "Turn on the Utility Light in the Utility Room on the Ground Floor.",
            "turnOn|Utility Light|*|*|*|Utility Room|Ground Floor",
            self.registry,
        )
        utility = store.prepare(
            actor_id="actor-a", session_id="session-a", grounded=utility_grounded,
            registry=self.registry, adapter=adapter,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            utility_future = pool.submit(
                store.commit, utility.confirmation(), registry=self.registry, adapter=adapter,
            )
            self.assertTrue(adapter.utility_started.wait(timeout=1))
            dependent_future = pool.submit(
                store.commit, dependent.confirmation(), registry=self.registry, adapter=adapter,
            )
            adapter.release_utility.set()
            utility_result = utility_future.result(timeout=2)
            dependent_result = dependent_future.result(timeout=2)
        self.assertEqual(utility_result.status, "COMMITTED")
        self.assertEqual((dependent_result.reason, len(adapter.sut_calls)), ("state_changed", 1))

    def test_b1_binds_plan_and_session_but_deliberately_omits_temporal_guards(self) -> None:
        adapter, store, action = self.prepare(ClarifyPrepareStore)
        rejected = store.commit(
            altered_confirmation(action.confirmation(), session_id="session-b"),
            registry=self.registry,
            adapter=adapter,
        )
        self.assertEqual((rejected.reason, len(adapter.sut_calls)), ("session_mismatch", 0))

        adapter, store, action = self.prepare(ClarifyPrepareStore)
        first = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        second = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertTrue(first.accepted and second.accepted)
        self.assertEqual(len(adapter.sut_calls), 2)

        adapter, store, action = self.prepare(ClarifyPrepareStore)
        adapter.mutate_state_for_setup("light.study_ceiling")
        drift = store.commit(action.confirmation(), registry=self.registry, adapter=adapter)
        self.assertTrue(drift.accepted)
        self.assertEqual(len(adapter.sut_calls), 1)


class _HAHandler(BaseHTTPRequestHandler):
    token = "test-token-not-a-real-secret"
    state = {"entity_id": "light.demo", "state": "off", "attributes": {"brightness": 0}}
    climate_state = {
        "entity_id": "climate.demo",
        "state": "cool",
        "attributes": {
            "temperature": 24.0,
            "hvac_modes": ["off", "cool"],
            "supported_features": 1,
        },
    }
    config: object = {"unit_system": {"temperature": "°C"}}
    get_paths: list[str] = []
    calls: list[dict[str, object]] = []
    post_status: int | None = None

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self._json(401, {})
            return
        type(self).get_paths.append(self.path)
        if self.path == "/api/states/light.demo":
            self._json(200, self.state)
        elif self.path == "/api/states/climate.demo":
            self._json(200, self.climate_state)
        elif self.path == "/api/config":
            self._json(200, self.config)
        else:
            self._json(404, {})

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self._json(401, {})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).calls.append(payload)
        if type(self).post_status is not None:
            self._json(type(self).post_status, {})
            return
        if self.path == "/api/services/light/turn_on":
            type(self).state = {
                "entity_id": "light.demo", "state": "on", "attributes": {"brightness": 0},
            }
            self._json(200, [type(self).state])
        else:
            self._json(404, {})


class RestAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _HAHandler.state = {"entity_id": "light.demo", "state": "off", "attributes": {"brightness": 0}}
        _HAHandler.climate_state = {
            "entity_id": "climate.demo",
            "state": "cool",
            "attributes": {
                "temperature": 24.0,
                "hvac_modes": ["off", "cool"],
                "supported_features": 1,
            },
        }
        _HAHandler.config = {"unit_system": {"temperature": "°C"}}
        _HAHandler.get_paths = []
        _HAHandler.calls = []
        _HAHandler.post_status = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _HAHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_official_rest_shape_and_loopback_guard(self) -> None:
        port = self.server.server_address[1]
        adapter = HomeAssistantRESTAdapter(
            f"http://127.0.0.1:{port}", _HAHandler.token, poll_seconds=0.2,
        )
        before = adapter.get_state("light.demo")
        after = adapter.call_service("light", "turn_on", {"entity_id": "light.demo"})
        self.assertEqual((before["state"], after.after["state"]), ("off", "on"))
        self.assertEqual(len(adapter.sut_calls), 1)
        with self.assertRaises(ValueError):
            HomeAssistantRESTAdapter("https://example.com", "token")

    def test_http_4xx_is_rejected_but_5xx_has_unknown_dispatch_outcome(self) -> None:
        port = self.server.server_address[1]
        adapter = HomeAssistantRESTAdapter(
            f"http://127.0.0.1:{port}", _HAHandler.token, poll_seconds=0.2,
        )
        for status, unknown, outcome in (
            (400, False, "request_rejected"),
            (500, True, "request_error_outcome_unknown"),
        ):
            with self.subTest(status=status):
                _HAHandler.post_status = status
                with self.assertRaises(ServiceCallError) as caught:
                    adapter.call_service("light", "turn_on", {"entity_id": "light.demo"})
                self.assertTrue(caught.exception.attempted)
                self.assertFalse(caught.exception.acknowledged)
                self.assertEqual(caught.exception.outcome_unknown, unknown)
                self.assertEqual(adapter.sut_calls[-1]["outcome"], outcome)

    def test_climate_temperature_unit_comes_from_official_config(self) -> None:
        port = self.server.server_address[1]
        adapter = HomeAssistantRESTAdapter(
            f"http://127.0.0.1:{port}", _HAHandler.token, poll_seconds=0.2,
        )
        state = adapter.get_state("climate.demo")
        self.assertEqual(state["attributes"]["temperature_unit"], "°C")
        self.assertNotIn("temperature_unit", _HAHandler.climate_state["attributes"])
        self.assertEqual(
            _HAHandler.get_paths,
            ["/api/states/climate.demo", "/api/config"],
        )

        malformed = (
            [],
            {},
            {"unit_system": {}},
            {"unit_system": {"temperature": 123}},
        )
        for config in malformed:
            with self.subTest(config=config):
                _HAHandler.config = config
                with self.assertRaises(AdapterError):
                    adapter.get_state("climate.demo")


if __name__ == "__main__":
    unittest.main()
