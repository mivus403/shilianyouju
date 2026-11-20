"""示例模式：读取预先保存的真实模型运行结果，并提供可一键套用的本人条件。

示例结果来自此前对 samples 中素材的真实接口调用，页面使用它们时不会再次调用模型。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preparation import (
    EquipmentAvailability,
    EquipmentUserReport,
    PreparationInput,
    SelectedEquipment,
    UserConditions,
    observation_id,
)


class DemoError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoPreset:
    preset_id: str
    label: str
    description: str
    scene_id: str | None
    preparation: PreparationInput


def demo_dir(project_root: Path) -> Path:
    return project_root / "samples" / "demo_results"


def list_cases(project_root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(demo_dir(project_root).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.append({"id": data["id"], "title": data["title"], "tagline": data["tagline"]})
    return cases


def load_case(project_root: Path, case_id: str) -> dict[str, Any]:
    path = demo_dir(project_root) / f"{case_id}.json"
    if not path.is_file():
        raise DemoError(f"没有找到示例案例 {case_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def scene_by_id(case: dict[str, Any], scene_id: str | None) -> dict[str, Any] | None:
    if not case["scenes"]:
        return None
    for scene in case["scenes"]:
        if scene["id"] == scene_id:
            return scene
    return case["scenes"][0]


def _equipment(scene_result: dict, roles: list[str], report: EquipmentUserReport) -> SelectedEquipment:
    candidate = scene_result["equipment_candidates"][0]
    return SelectedEquipment(
        id="eq-1",
        candidate_id=candidate["candidate_id"],
        type=candidate["type"],
        photo_ids=candidate["photo_ids"],
        roles=roles,
        user_report=report,
    )


def _conditions(**overrides: str) -> UserConditions:
    values = {
        "revision": 1,
        "current_discomfort": "no",
        "support_needed": "not_applicable",
        "restriction_status": "none_known",
        "sit_to_stand_assistance": "not_applicable",
        "helper_available": "not_applicable",
    }
    values.update(overrides)
    return UserConditions(**values)


def build_presets(case: dict[str, Any]) -> list[DemoPreset]:
    case_id = case["id"]
    if case_id == "A":
        none = EquipmentAvailability(seat="not_applicable", standing_support="not_applicable")
        return [
            DemoPreset("A1", "身体无不适、无已知限制", "最常见的情况：没有不适，也没有被告知的限制。",
                       None, PreparationInput(revision=1, area_name="客厅", user_conditions=_conditions(), equipment_availability=none)),
            DemoPreset("A2", "今天感觉有些不适", "当前有不适时，规则会提示先不要硬撑，这条不能被勾选掉。",
                       None, PreparationInput(revision=1, area_name="客厅", user_conditions=_conditions(current_discomfort="yes"), equipment_availability=none)),
        ]
    if case_id == "B":
        before = scene_by_id(case, "before")["run"]["result"]
        after = scene_by_id(case, "after")["run"]["result"]
        base_conditions = dict(support_needed="yes", sit_to_stand_assistance="yes", helper_available="no")
        availability = EquipmentAvailability(seat="yes", standing_support="yes")
        roles = ["seat", "standing_support"]
        b1 = PreparationInput(
            revision=1,
            area_name="客厅椅子前",
            user_conditions=_conditions(**base_conditions),
            equipment_availability=availability,
            selected_equipment=[_equipment(before, roles, EquipmentUserReport(has_wheels="yes", stability_confirmed="unknown", non_slip_confirmed="unknown"))],
            observation_confirmations={observation_id(before, 1): "confirmed"},
        )
        b2 = b1.model_copy(deep=True)
        b2.user_conditions.helper_available = "yes"
        b3 = PreparationInput(
            revision=1,
            area_name="客厅椅子前",
            user_conditions=_conditions(**{**base_conditions, "helper_available": "yes"}),
            equipment_availability=availability,
            selected_equipment=[_equipment(after, roles, EquipmentUserReport(has_wheels="no", stability_confirmed="yes", non_slip_confirmed="unknown"))],
            observation_confirmations={observation_id(after, 2): "rejected"},
        )
        return [
            DemoPreset("B1", "需要协助，但这次没人在场；椅子带轮", "使用“调整前”场地：协助缺口、带轮椅子、脚边小物品都会被提示。", "before", b1),
            DemoPreset("B2", "只把“有人协助”改成有", "其他不变：协助提示消失，椅子和小物品提示仍保留。", "before", b2),
            DemoPreset("B3", "换成无轮、稳固的椅子并移开物品", "使用“调整后”场地：旧提示被清除，防滑仍需本人确认，所以仍保留未知。", "after", b3),
        ]
    if case_id == "C":
        scene = scene_by_id(case, "unknown")["run"]["result"]
        return [
            DemoPreset(
                "C1",
                "多项信息都不确定",
                "不适、支撑需求、椅子是否稳固都选“不确定”，规则会保留未知，而不是替你当作没问题。",
                "unknown",
                PreparationInput(
                    revision=1,
                    area_name="客厅椅子旁",
                    user_conditions=_conditions(current_discomfort="unknown", support_needed="unknown", restriction_status="unknown"),
                    equipment_availability=EquipmentAvailability(seat="not_applicable", standing_support="yes"),
                    selected_equipment=[_equipment(scene, ["standing_support"], EquipmentUserReport(has_wheels="unknown", stability_confirmed="unknown", non_slip_confirmed="unknown"))],
                ),
            )
        ]
    return []
