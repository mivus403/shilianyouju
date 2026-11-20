"""示例模式：保存的真实模型结果能离线载入，并驱动本地规则和报告。"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.demo import build_presets, list_cases, load_case, scene_by_id
from src.knowledge import load_knowledge
from src.preparation import evaluate_preparation, preparation_to_serializable
from src.reporting import build_report

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = [item["id"] for item in list_cases(ROOT)]


def test_three_demo_cases_exist():
    assert CASE_IDS == ["A", "B", "C"]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_demo_case_matches_knowledge_and_media(case_id):
    knowledge = load_knowledge(ROOT)
    case = load_case(ROOT, case_id)
    assert case["claim_run"]["result"]["knowledge_version"] == knowledge.package.version
    assert (ROOT / "samples" / case["video_file"]).is_file()
    for scene in case["scenes"]:
        assert all((ROOT / "samples" / name).is_file() for name in scene["files"])


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_presets_run_locally_and_build_a_report(case_id):
    knowledge = load_knowledge(ROOT)
    case = load_case(ROOT, case_id)
    presets = build_presets(case)
    assert presets
    for preset in presets:
        scene = scene_by_id(case, preset.scene_id)
        run = evaluate_preparation(
            case["video_run"]["result"], scene["run"]["result"] if scene else None, preset.preparation, knowledge, revision=1
        )
        report = build_report(
            case["video_run"],
            scene["run"] if scene else None,
            case["claim_run"],
            preparation_to_serializable(run),
            knowledge,
            revision=1,
        )
        assert report.result.workflow_status in {"complete", "partial"}


def test_case_b_presets_change_the_rules_as_documented():
    knowledge = load_knowledge(ROOT)
    case = load_case(ROOT, "B")
    rules = {}
    for preset in build_presets(case):
        scene = scene_by_id(case, preset.scene_id)
        run = evaluate_preparation(case["video_run"]["result"], scene["run"]["result"], preset.preparation, knowledge, revision=1)
        rules[preset.preset_id] = {tuple(card.rule_ids): card.status for card in run.result.cards}
    assert rules["B1"][("R03",)] == "condition_gap"
    assert ("R03",) not in rules["B2"]
    assert rules["B2"][("R04", "R05")] == "condition_gap"
    assert rules["B3"][("R04", "R05")] == "needs_confirmation"


def test_app_demo_flow_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    next(button for button in app.button if "案例 B" in button.label).click()
    app.run()
    assert not app.exception
    assert app.session_state["claim_check_run"] and app.session_state["scene_run"]
    next(button for button in app.button if button.label.startswith("B1 ")).click()
    app.run()
    assert not app.exception
    assert app.session_state["preparation_run"]["model_requests"] == 0
    next(button for button in app.button if "生成或更新限定范围报告" in button.label).click()
    app.run()
    assert not app.exception
    assert app.session_state["report_run"]["result"]["workflow_status"] == "complete"
    next(button for button in app.button if "退出示例" in button.label).click()
    app.run()
    assert app.session_state["demo_case"] is None and app.session_state["video_run"] is None
