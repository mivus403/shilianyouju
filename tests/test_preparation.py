from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import AnalysisError
from src.knowledge import load_knowledge
from src.preparation import (
    EquipmentAvailability,
    EquipmentUserReport,
    PreparationInput,
    RestrictionMatch,
    SelectedEquipment,
    UserConditions,
    evaluate_preparation,
    observation_id,
    preparation_to_serializable,
    set_handling_state,
)


ROOT = Path(__file__).resolve().parents[1]


def video(action_type="sit_to_stand", *, recognition="clear", scope="supported", status="complete"):
    return {
        "analysis_id": "video-task5",
        "revision": 1,
        "status": status,
        "unprocessed_ranges": [[10.0, 12.0]] if status == "partial" else [],
        "actions": [
            {
                "id": "action-1",
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "type": action_type,
                "description": "测试动作",
                "equipment": ["chair"],
                "recognition": recognition,
                "scope": scope,
            }
        ],
    }


def scene(*, analysis_id="scene-before", wheels="present", view=False):
    observations = [
        {
            "photo_id": "photo-1",
            "equipment_candidate_id": "chair-actual",
            "kind": "visible_wheels" if wheels == "present" else "support_visible",
            "location": "椅子底部",
            "description": "当前器材候选",
            "recognition": "clear",
        },
        {
            "photo_id": "photo-2",
            "equipment_candidate_id": None,
            "kind": "possible_obstacle",
            "location": "椅子前方",
            "description": "蓝色小物品",
            "recognition": "clear",
        },
    ]
    if view:
        observations[0]["kind"] = "view_insufficient"
        observations[0]["recognition"] = "uncertain"
    return {
        "analysis_id": analysis_id,
        "revision": 1,
        "status": "complete",
        "equipment_candidates": [
            {
                "candidate_id": "chair-actual",
                "type": "chair",
                "photo_ids": ["photo-1"],
                "description": "前景椅子",
                "attributes": [
                    {
                        "name": "has_wheels",
                        "value": wheels,
                        "evidence": "底部视图",
                        "recognition": "uncertain" if wheels == "unknown" else "clear",
                    }
                ],
                "recognition": "uncertain" if wheels == "unknown" else "clear",
            },
            {
                "candidate_id": "chair-background",
                "type": "chair",
                "photo_ids": ["photo-2"],
                "description": "背景椅子",
                "attributes": [
                    {"name": "has_wheels", "value": "absent", "evidence": "底部可见", "recognition": "clear"}
                ],
                "recognition": "clear",
            },
        ],
        "observations": observations,
        "uncertainties": [],
    }


def preparation(
    *,
    helper="no",
    selected_candidate="chair-actual",
    roles=("seat", "standing_support"),
    wheels="yes",
    stability="unknown",
    non_slip="unknown",
    obstacle="confirmed",
    conditions=None,
    scene_result=None,
    restriction_matches=None,
):
    scene_result = scene_result or scene()
    observation_key = observation_id(scene_result, 2)
    candidate = next(item for item in scene_result["equipment_candidates"] if item["candidate_id"] == selected_candidate)
    selected = SelectedEquipment(
        id="eq-1",
        candidate_id=selected_candidate,
        type=candidate["type"],
        photo_ids=candidate["photo_ids"],
        roles=list(roles),
        user_report=EquipmentUserReport(
            has_wheels=wheels,
            stability_confirmed=stability,
            non_slip_confirmed=non_slip,
        ),
    )
    return PreparationInput(
        revision=1,
        area_name="客厅椅子前",
        user_conditions=conditions
        or UserConditions(
            revision=1,
            support_needed="yes",
            current_discomfort="no",
            restriction_status="none_known",
            sit_to_stand_assistance="yes",
            helper_available=helper,
        ),
        equipment_availability=EquipmentAvailability(seat="yes", standing_support="yes"),
        selected_equipment=[selected],
        observation_confirmations={observation_key: obstacle},
        restriction_matches=restriction_matches or [],
    )


def rule_cards(result, rule_id):
    return [card for card in result.cards if rule_id in card.rule_ids]


def test_b1_to_b2_only_helper_change_keeps_equipment_and_obstacle_cards():
    knowledge = load_knowledge(ROOT)
    scene_result = scene()
    b1 = evaluate_preparation(video(), scene_result, preparation(scene_result=scene_result), knowledge, revision=1)
    assert rule_cards(b1.result, "R03")[0].status == "condition_gap"
    assert rule_cards(b1.result, "R04")[0].status == "condition_gap"
    assert rule_cards(b1.result, "R05")[0].status == "condition_gap"
    assert rule_cards(b1.result, "R06")[0].status == "condition_gap"
    assert len([card for card in b1.result.cards if set(card.rule_ids) == {"R04", "R05"}]) == 1

    b2 = evaluate_preparation(
        video(),
        scene_result,
        preparation(helper="yes", scene_result=scene_result),
        knowledge,
        revision=2,
    )
    assert not rule_cards(b2.result, "R03")
    assert rule_cards(b2.result, "R04")
    assert rule_cards(b2.result, "R06")
    assert preparation_to_serializable(b2)["model_requests"] == 0


def test_b3_new_scene_does_not_accept_old_observation_and_keeps_non_slip_unknown():
    knowledge = load_knowledge(ROOT)
    before = scene()
    after = scene(analysis_id="scene-after", wheels="absent")
    old_observation = observation_id(before, 2)
    value = preparation(
        helper="yes",
        wheels="no",
        stability="yes",
        non_slip="unknown",
        obstacle="rejected",
        scene_result=after,
    )
    value.observation_confirmations = {old_observation: "confirmed"}
    with pytest.raises(AnalysisError, match="旧照片"):
        evaluate_preparation(video(), after, value, knowledge, revision=3)

    value.observation_confirmations = {observation_id(after, 2): "rejected"}
    run = evaluate_preparation(video(), after, value, knowledge, revision=3)
    equipment_card = rule_cards(run.result, "R04")[0]
    assert equipment_card.status == "needs_confirmation"
    assert any("滑动" in item for item in equipment_card.remaining_unknowns)
    assert not rule_cards(run.result, "R03")
    assert not rule_cards(run.result, "R06")


def test_c_unknown_body_and_obscured_support_remain_unknown():
    knowledge = load_knowledge(ROOT)
    scene_result = scene(analysis_id="scene-c", wheels="unknown", view=True)
    conditions = UserConditions(
        revision=1,
        support_needed="yes",
        current_discomfort="unknown",
        restriction_status="unknown",
        sit_to_stand_assistance="not_applicable",
        helper_available="not_applicable",
    )
    value = preparation(
        roles=("standing_support",),
        wheels="unknown",
        stability="unknown",
        non_slip="unknown",
        obstacle="unknown",
        conditions=conditions,
        scene_result=scene_result,
    )
    value.equipment_availability.seat = "not_applicable"
    run = evaluate_preparation(video("supported_calf_raise"), scene_result, value, knowledge, revision=1)
    assert rule_cards(run.result, "R01")[0].status == "needs_confirmation"
    assert rule_cards(run.result, "R02")[0].status == "needs_confirmation"
    support_card = rule_cards(run.result, "R05")[0]
    assert support_card.status == "needs_confirmation"
    assert support_card.evidence_ids == ["E05"]
    assert any("视角不足" in item for item in support_card.remaining_unknowns)
    assert rule_cards(run.result, "R06")[0].status == "needs_confirmation"


def test_photo_and_user_wheel_conflict_is_not_silently_resolved():
    knowledge = load_knowledge(ROOT)
    scene_result = scene(wheels="present")
    value = preparation(
        roles=("seat",),
        wheels="no",
        stability="yes",
        non_slip="yes",
        obstacle="rejected",
        scene_result=scene_result,
    )
    value.equipment_availability.standing_support = "not_applicable"
    value.user_conditions.support_needed = "no"
    run = evaluate_preparation(video(), scene_result, value, knowledge, revision=1)
    card = rule_cards(run.result, "R04")[0]
    assert card.status == "needs_confirmation"
    assert any("两种来源不一致" in item for item in card.remaining_unknowns)
    assert observation_id(scene_result, 1) in card.fact_refs


def test_only_selected_candidate_supplies_equipment_attributes():
    knowledge = load_knowledge(ROOT)
    scene_result = scene(wheels="present")
    value = preparation(
        selected_candidate="chair-background",
        roles=("seat",),
        wheels="no",
        stability="yes",
        non_slip="yes",
        obstacle="rejected",
        scene_result=scene_result,
    )
    value.equipment_availability.standing_support = "not_applicable"
    value.user_conditions.support_needed = "no"
    run = evaluate_preparation(video(), scene_result, value, knowledge, revision=1)
    assert not rule_cards(run.result, "R04")


def test_arm_raise_restriction_needs_confirmation_until_user_matches_it():
    knowledge = load_knowledge(ROOT)
    conditions = UserConditions(
        revision=1,
        support_needed="not_applicable",
        current_discomfort="no",
        restriction_status="has_restrictions",
        restrictions=["avoid_arm_raise"],
        sit_to_stand_assistance="not_applicable",
        helper_available="not_applicable",
    )
    value = PreparationInput(
        revision=1,
        user_conditions=conditions,
        equipment_availability=EquipmentAvailability(),
    )
    pending = evaluate_preparation(video("seated_arm_movement"), None, value, knowledge, revision=1)
    assert rule_cards(pending.result, "R02")[0].status == "needs_confirmation"

    value.restriction_matches = [
        RestrictionMatch(action_id="action-1", restriction="avoid_arm_raise", user_confirmation="yes")
    ]
    matched = evaluate_preparation(video("seated_arm_movement"), None, value, knowledge, revision=2)
    assert rule_cards(matched.result, "R02")[0].status == "condition_gap"


def test_body_rules_still_run_without_scene_and_partial_video_is_disclosed():
    knowledge = load_knowledge(ROOT)
    conditions = UserConditions(
        revision=1,
        support_needed="unknown",
        current_discomfort="yes",
        restriction_status="none_known",
        sit_to_stand_assistance="yes",
        helper_available="no",
    )
    value = PreparationInput(
        revision=1,
        user_conditions=conditions,
        equipment_availability=EquipmentAvailability(seat="unknown", standing_support="unknown"),
    )
    run = evaluate_preparation(video(status="partial"), None, value, knowledge, revision=1)
    assert run.result.status == "partial"
    assert rule_cards(run.result, "R01")[0].status == "condition_gap"
    assert rule_cards(run.result, "R03")[0].status == "condition_gap"
    assert not rule_cards(run.result, "R04")
    assert not rule_cards(run.result, "R05")
    assert run.result.checked_rule_ids == ["R01", "R02", "R03"]
    assert any("场地" in item and "尚未检查" in item for item in run.result.coverage_notes)
    assert any("未覆盖范围" in item for item in run.result.coverage_notes)


def test_unsupported_action_only_gets_scope_card():
    knowledge = load_knowledge(ROOT)
    value = PreparationInput(
        revision=1,
        user_conditions=UserConditions(revision=1),
        equipment_availability=EquipmentAvailability(),
    )
    run = evaluate_preparation(video("other", scope="unsupported"), None, value, knowledge, revision=1)
    assert len(run.result.cards) == 1
    assert run.result.cards[0].status == "unsupported"
    assert run.result.checked_rule_ids == []


def test_handling_state_does_not_change_gap_and_body_card_cannot_be_closed():
    knowledge = load_knowledge(ROOT)
    scene_result = scene()
    run = evaluate_preparation(video(), scene_result, preparation(scene_result=scene_result), knowledge, revision=1)
    serialized = preparation_to_serializable(run)
    equipment_card = next(card for card in serialized["result"]["cards"] if "R04" in card["rule_ids"])
    set_handling_state(serialized, equipment_card["id"], "user_reported_done")
    assert equipment_card["handling_state"] == "user_reported_done"
    assert equipment_card["status"] == "condition_gap"

    body = UserConditions(
        revision=1,
        current_discomfort="yes",
        restriction_status="none_known",
        support_needed="no",
        sit_to_stand_assistance="no",
        helper_available="not_applicable",
    )
    body_run = evaluate_preparation(
        video(),
        None,
        PreparationInput(
            revision=1,
            user_conditions=body,
            equipment_availability=EquipmentAvailability(seat="unknown", standing_support="not_applicable"),
        ),
        knowledge,
        revision=1,
    )
    body_data = preparation_to_serializable(body_run)
    body_card = next(card for card in body_data["result"]["cards"] if "R01" in card["rule_ids"])
    with pytest.raises(AnalysisError, match="不能用完成按钮"):
        set_handling_state(body_data, body_card["id"], "user_reported_done")
