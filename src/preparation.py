from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import AnalysisError
from .knowledge import KnowledgeBase


Answer = Literal["yes", "no", "unknown", "not_applicable"]
KnownAnswer = Literal["yes", "no", "unknown"]
Restriction = Literal[
    "avoid_repeated_sit_to_stand",
    "avoid_unsupported_standing",
    "avoid_calf_raise",
    "avoid_arm_raise",
]
Confirmation = Literal["confirmed", "rejected", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserConditions(StrictModel):
    revision: int = Field(ge=1)
    support_needed: Answer = "unknown"
    current_discomfort: Answer = "unknown"
    restriction_status: Literal["none_known", "has_restrictions", "unknown"] = "unknown"
    restrictions: list[Restriction] = Field(default_factory=list)
    restriction_note: str | None = None
    sit_to_stand_assistance: Answer = "not_applicable"
    helper_available: Answer = "not_applicable"

    @model_validator(mode="after")
    def validate_restrictions(self):
        if len(self.restrictions) != len(set(self.restrictions)):
            raise ValueError("限制标签不能重复")
        if self.restriction_status != "has_restrictions" and (self.restrictions or self.restriction_note):
            raise ValueError("只有确认存在活动限制时才能填写限制内容")
        return self


class EquipmentAvailability(StrictModel):
    seat: Answer = "not_applicable"
    standing_support: Answer = "not_applicable"


class EquipmentUserReport(StrictModel):
    has_wheels: KnownAnswer = "unknown"
    stability_confirmed: KnownAnswer = "unknown"
    non_slip_confirmed: KnownAnswer = "unknown"


class SelectedEquipment(StrictModel):
    id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    photo_ids: list[str] = Field(min_length=1)
    roles: list[Literal["seat", "standing_support"]] = Field(min_length=1)
    user_report: EquipmentUserReport

    @model_validator(mode="after")
    def validate_unique_values(self):
        if len(self.photo_ids) != len(set(self.photo_ids)):
            raise ValueError("器材照片 ID 不能重复")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("器材用途不能重复")
        return self


class RestrictionMatch(StrictModel):
    action_id: str = Field(min_length=1)
    restriction: Restriction | Literal["other_note"]
    user_confirmation: KnownAnswer = "unknown"


class PreparationInput(StrictModel):
    revision: int = Field(ge=1)
    area_name: str | None = None
    user_conditions: UserConditions
    equipment_availability: EquipmentAvailability
    selected_equipment: list[SelectedEquipment] = Field(default_factory=list)
    observation_confirmations: dict[str, Confirmation] = Field(default_factory=dict)
    restriction_matches: list[RestrictionMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_values(self):
        ids = [item.id for item in self.selected_equipment]
        if len(ids) != len(set(ids)):
            raise ValueError("本次器材 ID 不能重复")
        candidate_ids = [item.candidate_id for item in self.selected_equipment]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("同一个场地候选应合并用途，不能重复绑定")
        roles = [role for item in self.selected_equipment for role in item.roles]
        if len(roles) != len(set(roles)):
            raise ValueError("每种用途只能指定一件实际器材")
        match_keys = [(item.action_id, item.restriction) for item in self.restriction_matches]
        if len(match_keys) != len(set(match_keys)):
            raise ValueError("同一动作与限制不能重复确认")
        return self


class PreparationCard(StrictModel):
    id: str
    revision: int = Field(ge=1)
    category: Literal["preparation"] = "preparation"
    status: Literal["condition_gap", "needs_confirmation", "unsupported"]
    title: str = Field(min_length=1)
    rule_ids: list[str]
    fact_refs: list[str]
    condition_refs: list[str]
    evidence_ids: list[str]
    explanation: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    remaining_unknowns: list[str]
    handling_state: Literal["open", "user_reported_done", "uncertain"] = "open"
    can_mark_done: bool


class PreparationResult(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str
    revision: int = Field(ge=1)
    video_analysis_id: str
    scene_analysis_id: str | None
    knowledge_version: str
    status: Literal["complete", "partial", "failed"]
    checked_rule_ids: list[str]
    cards: list[PreparationCard]
    coverage_notes: list[str]


@dataclass(frozen=True)
class PreparationRun:
    result: PreparationResult
    input_snapshot: PreparationInput
    elapsed_seconds: float


@dataclass
class _Draft:
    rule_ids: list[str]
    status: str
    title: str
    fact_refs: list[str]
    condition_refs: list[str]
    evidence_ids: list[str]
    explanation: str
    suggested_action: str
    remaining_unknowns: list[str]
    can_mark_done: bool
    equipment_ids: list[str]


def observation_id(scene_result: dict[str, Any], index: int) -> str:
    return f"{scene_result['analysis_id']}:r{scene_result['revision']}:obs:{index}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _branch(knowledge: KnowledgeBase, rule_id: str, index: int) -> dict[str, Any]:
    try:
        return knowledge.condition_rules[rule_id]["branches"][index]
    except (KeyError, IndexError, TypeError) as exc:
        raise AnalysisError(f"本地规则 {rule_id} 结构无效") from exc


def _evidence(knowledge: KnowledgeBase, rule_id: str, *, override: list[str] | None = None) -> list[str]:
    evidence_ids = list(knowledge.condition_rules[rule_id].get("evidence_ids", [])) if override is None else override
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in knowledge.cards]
    if missing:
        raise AnalysisError(f"本地规则 {rule_id} 引用了不存在的资料：{missing}")
    return evidence_ids


def _make_draft(
    knowledge: KnowledgeBase,
    rule_id: str,
    branch_index: int,
    *,
    fact_refs: list[str],
    condition_refs: list[str],
    explanation: str,
    remaining_unknowns: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    equipment_ids: list[str] | None = None,
) -> _Draft:
    branch = _branch(knowledge, rule_id, branch_index)
    branch_evidence = branch.get("evidence_ids")
    resolved_evidence = _evidence(
        knowledge,
        rule_id,
        override=evidence_ids if evidence_ids is not None else branch_evidence,
    )
    return _Draft(
        rule_ids=[rule_id],
        status=branch["status"],
        title=branch["title"],
        fact_refs=_unique(fact_refs),
        condition_refs=_unique(condition_refs),
        evidence_ids=_unique(resolved_evidence),
        explanation=explanation,
        suggested_action=branch["suggested_action"],
        remaining_unknowns=_unique(remaining_unknowns or []),
        can_mark_done=bool(knowledge.condition_rules[rule_id].get("can_mark_done")),
        equipment_ids=_unique(equipment_ids or []),
    )


def _candidate_wheel_value(candidate: dict[str, Any]) -> str | None:
    for attribute in candidate.get("attributes", []):
        if attribute.get("name", "").lower() in {"has_wheels", "wheels", "visible_wheels"}:
            return attribute.get("value")
    return None


def _validate_current_refs(
    video_result: dict[str, Any],
    scene_result: dict[str, Any] | None,
    preparation: PreparationInput,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    action_lookup = {item["id"]: item for item in video_result.get("actions", [])}
    candidate_lookup: dict[str, dict[str, Any]] = {}
    observation_lookup: dict[str, dict[str, Any]] = {}
    if scene_result and scene_result.get("status") == "complete":
        candidate_lookup = {item["candidate_id"]: item for item in scene_result.get("equipment_candidates", [])}
        for index, item in enumerate(scene_result.get("observations", []), start=1):
            observation_lookup[observation_id(scene_result, index)] = item

    for equipment in preparation.selected_equipment:
        candidate = candidate_lookup.get(equipment.candidate_id)
        if candidate is None:
            raise AnalysisError(f"实际器材绑定了当前场地不存在的候选：{equipment.candidate_id}")
        if equipment.type != candidate.get("type") or equipment.photo_ids != candidate.get("photo_ids"):
            raise AnalysisError(f"实际器材 {equipment.id} 与当前场地候选内容不一致")
    invalid_observations = set(preparation.observation_confirmations) - set(observation_lookup)
    if invalid_observations:
        raise AnalysisError("场地确认引用了旧照片或不存在的观察项：" + "、".join(sorted(invalid_observations)))
    for match in preparation.restriction_matches:
        if match.action_id not in action_lookup:
            raise AnalysisError(f"限制确认引用了不存在的动作：{match.action_id}")
        allowed = set(preparation.user_conditions.restrictions)
        if preparation.user_conditions.restriction_note:
            allowed.add("other_note")
        if match.restriction not in allowed:
            raise AnalysisError(f"限制确认引用了未填写的限制：{match.restriction}")
    return candidate_lookup, observation_lookup


def _selected_for(preparation: PreparationInput, role: str) -> SelectedEquipment | None:
    return next((item for item in preparation.selected_equipment if role in item.roles), None)


def _related_observation_ids(
    observation_lookup: dict[str, dict[str, Any]],
    candidate_id: str,
    kinds: set[str] | None = None,
) -> list[str]:
    return [
        item_id
        for item_id, item in observation_lookup.items()
        if item.get("equipment_candidate_id") == candidate_id and (kinds is None or item.get("kind") in kinds)
    ]


def _rule_r01(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
) -> list[_Draft]:
    value = preparation.user_conditions.current_discomfort
    refs = [item["id"] for item in actions]
    if value == "yes":
        return [_make_draft(
            knowledge,
            "R01",
            0,
            fact_refs=refs,
            condition_refs=["user_conditions.current_discomfort"],
            explanation="本人填写目前有不适。该身体条件不会被场地调整或完成勾选消除。",
        )]
    if value in {"unknown", "not_applicable"}:
        return [_make_draft(
            knowledge,
            "R01",
            1,
            fact_refs=refs,
            condition_refs=["user_conditions.current_discomfort"],
            explanation="当前身体感受填写为不确定，规则保留未知，不预设为没有不适。",
        )]
    return []


def _rule_r02(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
) -> list[_Draft]:
    conditions = preparation.user_conditions
    refs = [item["id"] for item in actions]
    if conditions.restriction_status == "unknown":
        return [_make_draft(
            knowledge,
            "R02",
            0,
            fact_refs=refs,
            condition_refs=["user_conditions.restriction_status"],
            explanation="是否曾被明确告知活动限制尚未确认；本应用不会根据年龄或病名猜测。",
        )]
    if conditions.restriction_status == "none_known":
        return []
    if not conditions.restrictions and not (conditions.restriction_note or "").strip():
        return [_make_draft(
            knowledge, "R02", 2,
            fact_refs=refs,
            condition_refs=["user_conditions.restriction_status"],
            explanation="本人填写存在活动限制，但未提供可对照的内容，不能当成没有限制。",
            remaining_unknowns=["原有活动限制的具体内容"],
        )]

    match_lookup = {
        (item.action_id, item.restriction): item.user_confirmation
        for item in preparation.restriction_matches
    }
    relevant_types = {
        "avoid_repeated_sit_to_stand": {"sit_to_stand"},
        "avoid_unsupported_standing": {"sit_to_stand", "supported_calf_raise"},
        "avoid_calf_raise": {"supported_calf_raise"},
        "avoid_arm_raise": {"seated_arm_movement"},
        "other_note": {"seated_arm_movement", "sit_to_stand", "supported_calf_raise"},
    }
    gaps: list[str] = []
    unknowns: list[str] = []
    condition_refs = ["user_conditions.restriction_status", "user_conditions.restrictions"]
    restrictions: list[str] = list(conditions.restrictions)
    if conditions.restriction_note:
        restrictions.append("other_note")
        condition_refs.append("user_conditions.restriction_note")
    for restriction in restrictions:
        for action in actions:
            if action["type"] not in relevant_types[restriction]:
                continue
            label = f"{action['id']} 与 {restriction} 的对应关系"
            if restriction == "avoid_calf_raise" and action["type"] == "supported_calf_raise":
                gaps.append(action["id"])
                continue
            confirmation = match_lookup.get((action["id"], restriction), "unknown")
            if confirmation == "yes":
                gaps.append(action["id"])
            elif confirmation == "unknown":
                unknowns.append(label + "尚未确认")
    if gaps:
        if unknowns:
            condition_refs.append("restriction_matches")
        return [_make_draft(
            knowledge,
            "R02",
            1,
            fact_refs=_unique(gaps + refs if conditions.restriction_note else gaps),
            condition_refs=condition_refs,
            explanation="已填写的活动限制与所列动作直接匹配，或由本人确认涉及该动作；未明对应关系仍单独保留。",
            remaining_unknowns=unknowns,
        )]
    if unknowns:
        return [_make_draft(
            knowledge,
            "R02",
            2,
            fact_refs=refs,
            condition_refs=condition_refs + ["restriction_matches"],
            explanation="动作标签不足以证明限制所指的动作细节，备注也不会由模型解释为新的禁忌。",
            remaining_unknowns=unknowns,
        )]
    return []


def _rule_r03(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
) -> list[_Draft]:
    sit_refs = [item["id"] for item in actions if item["type"] == "sit_to_stand"]
    if not sit_refs:
        return []
    assistance = preparation.user_conditions.sit_to_stand_assistance
    helper = preparation.user_conditions.helper_available
    condition_refs = ["user_conditions.sit_to_stand_assistance"]
    if assistance == "yes" and helper == "no":
        return [_make_draft(
            knowledge,
            "R03",
            0,
            fact_refs=sit_refs,
            condition_refs=condition_refs + ["user_conditions.helper_available"],
            explanation="本人填写坐站需要协助，但本次没有可协助的人；本规则不评价协助者能力。",
        )]
    if assistance in {"unknown", "not_applicable"} or (assistance == "yes" and helper in {"unknown", "not_applicable"}):
        return [_make_draft(
            knowledge,
            "R03",
            1,
            fact_refs=sit_refs,
            condition_refs=condition_refs + (["user_conditions.helper_available"] if assistance == "yes" else []),
            explanation="坐站是否需要协助或本次协助安排仍有未填写项，视频中的提醒不能替代本人安排。",
        )]
    return []


def _equipment_condition_refs(equipment: SelectedEquipment) -> list[str]:
    prefix = f"selected_equipment[{equipment.id}].user_report"
    return [
        f"{prefix}.has_wheels",
        f"{prefix}.stability_confirmed",
        f"{prefix}.non_slip_confirmed",
    ]


def _rule_r04(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
    candidate_lookup: dict[str, dict[str, Any]],
    observation_lookup: dict[str, dict[str, Any]],
) -> list[_Draft]:
    sit_refs = [item["id"] for item in actions if item["type"] == "sit_to_stand"]
    if not sit_refs:
        return []
    availability_ref = "equipment_availability.seat"
    availability = preparation.equipment_availability.seat
    if availability == "no":
        return [_make_draft(
            knowledge,
            "R04",
            0,
            fact_refs=sit_refs,
            condition_refs=[availability_ref],
            explanation="本人明确填写本次坐站没有准备座椅；这不是从照片未看见座椅推断的。",
        )]
    equipment = _selected_for(preparation, "seat")
    if availability in {"unknown", "not_applicable"} or equipment is None:
        unknowns = ["是否有本次坐站要使用的座椅尚未确认"] if availability != "yes" else ["尚未从当前场地候选中指定本次座椅"]
        return [_make_draft(
            knowledge,
            "R04",
            2,
            fact_refs=sit_refs,
            condition_refs=[availability_ref] + (["selected_equipment"] if equipment is None else []),
            explanation="照片里出现椅子不等于它就是本次实际座椅，必须由本人指定用途。",
            remaining_unknowns=unknowns,
        )]

    report = equipment.user_report
    candidate = candidate_lookup[equipment.candidate_id]
    photo_wheels = _candidate_wheel_value(candidate)
    obs_ids = _related_observation_ids(
        observation_lookup,
        equipment.candidate_id,
        {"visible_wheels", "view_insufficient"},
    )
    gaps: list[str] = []
    unknowns: list[str] = []
    if report.has_wheels == "yes":
        gaps.append("本人确认这把座椅有轮")
    if report.stability_confirmed == "no":
        gaps.append("本人确认这把座椅不稳固")
    if report.non_slip_confirmed == "no":
        gaps.append("本人确认这把座椅在本次摆放表面会滑动")
    if report.has_wheels == "unknown":
        unknowns.append("座椅是否有轮尚未由本人确认")
    if report.stability_confirmed == "unknown":
        unknowns.append("座椅稳固性尚未由本人确认，照片不能认证")
    if report.non_slip_confirmed == "unknown":
        unknowns.append("座椅在本次摆放表面是否滑动尚未由本人确认")
    if photo_wheels == "present" and report.has_wheels == "no":
        unknowns.append("照片候选显示轮状部件，但本人填写无轮；两种来源不一致")
    elif photo_wheels == "absent" and report.has_wheels == "yes":
        unknowns.append("照片候选未见轮状部件，但本人填写有轮；两种来源不一致")
    elif photo_wheels == "present" and report.has_wheels == "unknown":
        unknowns.append("照片候选显示轮状部件，仍需确认是否为当前实际座椅及其状态")
    if any(observation_lookup[item_id].get("kind") == "view_insufficient" for item_id in obs_ids):
        unknowns.append("当前照片对座椅关键部位的视角不足")

    refs = sit_refs + obs_ids
    condition_refs = [availability_ref] + _equipment_condition_refs(equipment)
    explanation = (
        f"本人指定 {equipment.id}（场地候选 {equipment.candidate_id}）作为本次坐站座椅。"
        "稳固性和防滑只采用本人确认；照片中的轮子信息只作为需核对的候选。"
    )
    if gaps:
        return [_make_draft(
            knowledge,
            "R04",
            1,
            fact_refs=refs,
            condition_refs=condition_refs,
            explanation=explanation + " 已确认的条件缺口：" + "；".join(gaps) + "。",
            remaining_unknowns=unknowns,
            equipment_ids=[equipment.id],
        )]
    if unknowns:
        return [_make_draft(
            knowledge,
            "R04",
            2,
            fact_refs=refs,
            condition_refs=condition_refs,
            explanation=explanation,
            remaining_unknowns=unknowns,
            equipment_ids=[equipment.id],
        )]
    return []


def _rule_r05(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
    observation_lookup: dict[str, dict[str, Any]],
) -> list[_Draft]:
    calf_refs = [item["id"] for item in actions if item["type"] == "supported_calf_raise"]
    sit_refs = [item["id"] for item in actions if item["type"] == "sit_to_stand"]
    support_answer = preparation.user_conditions.support_needed
    requires_support = bool(calf_refs) or (bool(sit_refs) and support_answer in {"yes", "unknown", "not_applicable"})
    if not requires_support:
        return []
    refs = calf_refs + (sit_refs if support_answer != "no" else [])
    evidence_ids = ["E05"] if calf_refs else []
    condition_refs = ["equipment_availability.standing_support"]
    if sit_refs:
        condition_refs.append("user_conditions.support_needed")
    if sit_refs and not calf_refs and support_answer in {"unknown", "not_applicable"}:
        return [_make_draft(
            knowledge,
            "R05",
            2,
            fact_refs=sit_refs,
            condition_refs=condition_refs,
            explanation="坐站动作标签不证明全程是否需要支撑，本人支撑需求仍未确认。",
            remaining_unknowns=["站立或坐站时是否需要支撑"],
            evidence_ids=[],
        )]

    availability = preparation.equipment_availability.standing_support
    if availability == "no":
        return [_make_draft(
            knowledge,
            "R05",
            0,
            fact_refs=refs,
            condition_refs=condition_refs,
            explanation="本次动作需要支撑，但本人明确填写尚未准备实际支撑。",
            evidence_ids=evidence_ids,
        )]
    equipment = _selected_for(preparation, "standing_support")
    if availability in {"unknown", "not_applicable"} or equipment is None:
        return [_make_draft(
            knowledge,
            "R05",
            2,
            fact_refs=refs,
            condition_refs=condition_refs + (["selected_equipment"] if equipment is None else []),
            explanation="照片中看见椅背或家具不等于已经指定并确认本次可靠支撑。",
            remaining_unknowns=["本次实际支撑及其状态尚未完整确认"],
            evidence_ids=evidence_ids,
        )]

    report = equipment.user_report
    obs_ids = _related_observation_ids(
        observation_lookup,
        equipment.candidate_id,
        {"visible_wheels", "support_visible", "view_insufficient"},
    )
    gaps: list[str] = []
    unknowns: list[str] = []
    if report.stability_confirmed == "no":
        gaps.append("本人确认所选支撑不稳固")
    if report.non_slip_confirmed == "no":
        gaps.append("本人确认所选支撑在本次摆放表面会滑动")
    if report.stability_confirmed == "unknown":
        unknowns.append("支撑稳固性尚未由本人确认")
    if report.non_slip_confirmed == "unknown":
        unknowns.append("支撑在本次摆放表面是否滑动尚未由本人确认")
    if any(observation_lookup[item_id].get("kind") == "visible_wheels" for item_id in obs_ids):
        unknowns.append("照片候选可见轮状部件；对支撑用途只提示核实实际稳固与防滑，不直接判定违规")
    if any(observation_lookup[item_id].get("kind") == "view_insufficient" for item_id in obs_ids):
        unknowns.append("当前照片对所选支撑关键部位的视角不足")
    condition_refs.extend(_equipment_condition_refs(equipment)[1:])
    explanation = (
        f"本人指定 {equipment.id}（场地候选 {equipment.candidate_id}）作为本次站立支撑。"
        "照片不能证明稳固、承重或防滑。"
    )
    if gaps:
        return [_make_draft(
            knowledge,
            "R05",
            1,
            fact_refs=refs + obs_ids,
            condition_refs=condition_refs,
            explanation=explanation + " 已确认的条件缺口：" + "；".join(gaps) + "。",
            remaining_unknowns=unknowns,
            evidence_ids=evidence_ids,
            equipment_ids=[equipment.id],
        )]
    if unknowns:
        return [_make_draft(
            knowledge,
            "R05",
            2,
            fact_refs=refs + obs_ids,
            condition_refs=condition_refs,
            explanation=explanation,
            remaining_unknowns=unknowns,
            evidence_ids=evidence_ids,
            equipment_ids=[equipment.id],
        )]
    return []


def _rule_r06(
    knowledge: KnowledgeBase,
    actions: list[dict[str, Any]],
    preparation: PreparationInput,
    observation_lookup: dict[str, dict[str, Any]],
) -> list[_Draft]:
    confirmed: list[str] = []
    unknown: list[str] = []
    for item_id, item in observation_lookup.items():
        kind = item.get("kind")
        if kind not in {"possible_obstacle", "view_insufficient"}:
            continue
        confirmation = preparation.observation_confirmations.get(item_id, "unknown")
        if confirmation == "confirmed" and kind == "possible_obstacle":
            confirmed.append(item_id)
        elif confirmation == "unknown" or (confirmation == "confirmed" and kind == "view_insufficient"):
            unknown.append(item_id)
    action_refs = [item["id"] for item in actions]
    condition_refs = [f"observation_confirmations[{item_id}]" for item_id in confirmed + unknown]
    if confirmed:
        return [_make_draft(
            knowledge,
            "R06",
            0,
            fact_refs=action_refs + confirmed + unknown,
            condition_refs=condition_refs,
            explanation="本人确认照片中的疑似小物品位于本次活动位置或进出路线；未明观察仍保留。",
            remaining_unknowns=[f"观察 {item_id} 与实际活动位置的关系仍未确认" for item_id in unknown],
        )]
    if unknown:
        return [_make_draft(
            knowledge,
            "R06",
            1,
            fact_refs=action_refs + unknown,
            condition_refs=condition_refs,
            explanation="当前照片存在疑似障碍或视角遮挡，但位置关系尚未由本人确认。",
            remaining_unknowns=[f"观察 {item_id} 与实际活动位置的关系" for item_id in unknown],
        )]
    return []


def _combine_equipment_cards(drafts: list[_Draft]) -> list[_Draft]:
    r04 = next((item for item in drafts if item.rule_ids == ["R04"] and item.equipment_ids), None)
    r05 = next((item for item in drafts if item.rule_ids == ["R05"] and item.equipment_ids), None)
    if not r04 or not r05 or not set(r04.equipment_ids) & set(r05.equipment_ids):
        return drafts
    merged = _Draft(
        rule_ids=["R04", "R05"],
        status="condition_gap" if "condition_gap" in {r04.status, r05.status} else "needs_confirmation",
        title="同一件座椅的坐站与支撑条件需要一起处理",
        fact_refs=_unique(r04.fact_refs + r05.fact_refs),
        condition_refs=_unique(r04.condition_refs + r05.condition_refs),
        evidence_ids=_unique(r04.evidence_ids + r05.evidence_ids),
        explanation=r04.explanation + " " + r05.explanation,
        suggested_action=r04.suggested_action + " " + r05.suggested_action,
        remaining_unknowns=_unique(r04.remaining_unknowns + r05.remaining_unknowns),
        can_mark_done=True,
        equipment_ids=_unique(r04.equipment_ids + r05.equipment_ids),
    )
    return [item for item in drafts if item is not r04 and item is not r05] + [merged]


def _card_id(
    video_result: dict[str, Any],
    scene_result: dict[str, Any] | None,
    preparation: PreparationInput,
    draft: _Draft,
) -> str:
    stable = {
        "video": [video_result.get("analysis_id"), video_result.get("revision")],
        "scene": [scene_result.get("analysis_id"), scene_result.get("revision")] if scene_result else None,
        "input_revision": preparation.revision,
        "rules": draft.rule_ids,
        "facts": draft.fact_refs,
        "conditions": draft.condition_refs,
    }
    digest = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"prep-card-{digest}"


def evaluate_preparation(
    video_result: dict[str, Any],
    scene_result: dict[str, Any] | None,
    preparation: PreparationInput,
    knowledge: KnowledgeBase,
    *,
    revision: int,
) -> PreparationRun:
    started = time.perf_counter()
    if video_result.get("status") == "failed":
        result = PreparationResult(
            analysis_id=f"preparation-{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]}",
            revision=revision,
            video_analysis_id=video_result.get("analysis_id", "unknown"),
            scene_analysis_id=scene_result.get("analysis_id") if scene_result else None,
            knowledge_version=knowledge.package.version,
            status="failed",
            checked_rule_ids=[],
            cards=[],
            coverage_notes=["视频提取失败，不能依据该视频的动作运行准备规则。"],
        )
        return PreparationRun(result, preparation, time.perf_counter() - started)

    candidate_lookup, observation_lookup = _validate_current_refs(video_result, scene_result, preparation)
    supported_actions: list[dict[str, Any]] = []
    drafts: list[_Draft] = []
    coverage_notes: list[str] = []
    for action in video_result.get("actions", []):
        if action.get("scope") == "unsupported" or action.get("type") == "other":
            drafts.append(_Draft(
                rule_ids=[],
                status="unsupported",
                title="该动作不在首版准备检查范围",
                fact_refs=[action["id"]],
                condition_refs=[],
                evidence_ids=[],
                explanation="首版只处理坐姿上肢活动、椅子坐站转换和扶椅提踵；不为范围外动作给执行或替代方案。",
                suggested_action="只保留范围提示；如需判断该动作，请使用适用于该动作的其他资料或服务。",
                remaining_unknowns=[],
                can_mark_done=False,
                equipment_ids=[],
            ))
        elif action.get("scope") != "supported" or action.get("recognition") != "clear" or action.get("type") == "unknown":
            drafts.append(_Draft(
                rule_ids=[],
                status="needs_confirmation",
                title="动作识别需要确认",
                fact_refs=[action["id"]],
                condition_refs=[],
                evidence_ids=[],
                explanation="动作类型或识别清晰度不足，当前不运行依赖该动作类型的准备规则。",
                suggested_action="先回看片段并确认动作；不能根据模糊识别给准备结论。",
                remaining_unknowns=["动作类型或动作范围"],
                can_mark_done=False,
                equipment_ids=[],
            ))
        else:
            supported_actions.append(action)

    if video_result.get("status") == "partial":
        coverage_notes.append("视频仅部分处理；内容提取中的不确定信息仍需确认。")
        if video_result.get("unprocessed_ranges"):
            coverage_notes.append(f"未覆盖范围（未处理时间）：{video_result['unprocessed_ranges']}。")
    else:
        coverage_notes.append("视频状态 complete 只表示请求和结构完整，不证明所有动作、声音或画面均已识别。")
    if not supported_actions:
        coverage_notes.append("没有清晰且在首版范围内的动作可运行 R01—R06。")
    else:
        drafts.extend(_rule_r01(knowledge, supported_actions, preparation))
        drafts.extend(_rule_r02(knowledge, supported_actions, preparation))
        drafts.extend(_rule_r03(knowledge, supported_actions, preparation))

    scene_complete = bool(scene_result and scene_result.get("status") == "complete")
    if supported_actions and scene_complete:
        drafts.extend(_rule_r04(knowledge, supported_actions, preparation, candidate_lookup, observation_lookup))
        drafts.extend(_rule_r05(knowledge, supported_actions, preparation, observation_lookup))
        drafts.extend(_rule_r06(knowledge, supported_actions, preparation, observation_lookup))
    if scene_result is None:
        coverage_notes.append("尚未提交场地提取结果；R01—R03可按本人条件检查，R04—R06的照片分支尚未检查。")
    elif not scene_complete:
        coverage_notes.append("场地提取失败；本人条件仍可检查，但不能据此判断实际器材或活动位置。")
    else:
        coverage_notes.append("场地照片只提供候选观察；稳固、承重和防滑仍以本人明确确认为准。")

    drafts = _combine_equipment_cards(drafts)
    cards = [
        PreparationCard(
            id=_card_id(video_result, scene_result, preparation, draft),
            revision=revision,
            status=draft.status,
            title=draft.title,
            rule_ids=draft.rule_ids,
            fact_refs=draft.fact_refs,
            condition_refs=draft.condition_refs,
            evidence_ids=draft.evidence_ids,
            explanation=draft.explanation,
            suggested_action=draft.suggested_action,
            remaining_unknowns=draft.remaining_unknowns,
            can_mark_done=draft.can_mark_done,
        )
        for draft in drafts
    ]
    checked_rules = ["R01", "R02", "R03"] if supported_actions else []
    if supported_actions and scene_complete:
        checked_rules.extend(["R04", "R05", "R06"])
    if not cards:
        coverage_notes.append("当前填写与已识别范围未触发提示；这不等于个人适练或场地获得安全许可。")
    partial = video_result.get("status") == "partial" or not scene_complete
    result = PreparationResult(
        analysis_id=f"preparation-{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]}",
        revision=revision,
        video_analysis_id=video_result["analysis_id"],
        scene_analysis_id=scene_result.get("analysis_id") if scene_result else None,
        knowledge_version=knowledge.package.version,
        status="partial" if partial else "complete",
        checked_rule_ids=checked_rules,
        cards=cards,
        coverage_notes=coverage_notes,
    )
    return PreparationRun(result, preparation, time.perf_counter() - started)


def preparation_to_serializable(run: PreparationRun) -> dict[str, Any]:
    return {
        "result": run.result.model_dump(mode="json"),
        "input_snapshot": run.input_snapshot.model_dump(mode="json"),
        "elapsed_seconds": run.elapsed_seconds,
        "model_requests": 0,
    }


def set_handling_state(run_data: dict[str, Any], card_id: str, state: str) -> None:
    if state not in {"open", "user_reported_done", "uncertain"}:
        raise AnalysisError("无效的处理状态")
    for card in run_data.get("result", {}).get("cards", []):
        if card.get("id") != card_id:
            continue
        if not card.get("can_mark_done"):
            raise AnalysisError("身体条件或识别卡不能用完成按钮消除")
        card["handling_state"] = state
        return
    raise AnalysisError("找不到对应的准备检查卡")
