from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import AnalysisError
from src.knowledge import load_knowledge
from src.reporting import build_report, report_to_serializable


ROOT = Path(__file__).resolve().parents[1]


def video_run(*, status="complete", analysis_id="video-report"):
    return {
        "result": {
            "schema_version": "0.1",
            "analysis_id": analysis_id,
            "revision": 2,
            "video_id": "video-hash",
            "duration_seconds": 25.0,
            "audio_status": "present",
            "status": status,
            "unprocessed_ranges": [[20.0, 25.0]] if status == "partial" else [],
            "actions": [
                {
                    "id": "action-1",
                    "start_seconds": 1.0,
                    "end_seconds": 18.0,
                    "type": "sit_to_stand",
                    "description": "人物从椅子站起再坐下",
                    "equipment": ["chair"],
                    "recognition": "clear",
                    "scope": "supported",
                }
            ],
            "claims": [],
            "cautions": [],
            "uncertainties": ["不能证明全部声音均已识别"],
        },
        "raw_responses": [{"text": "不应进入报告"}],
        "requests": [
            {
                "label": "视频 1/1",
                "elapsed_seconds": 12.5,
                "request_id": "request-video",
                "response_model": "qwen-test",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        ],
    }


def scene_run(analysis_id="scene-report"):
    return {
        "result": {
            "schema_version": "0.1",
            "analysis_id": analysis_id,
            "revision": 3,
            "status": "complete",
            "photos": [{"id": "photo-1", "role": "extra", "path": "photo.bin"}],
            "selected_equipment": [],
            "visible_items": [],
            "equipment_candidates": [],
            "observations": [],
            "uncertainties": ["照片不能证明稳固性"],
        },
        "raw_responses": [{"text": "不应进入报告"}],
        "requests": [],
    }


def claim_run(*, video_analysis_id="video-report", evidence_id="E01", statement="持续不适也要硬撑"):
    return {
        "result": {
            "schema_version": "0.1",
            "analysis_id": "claim-report",
            "revision": 1,
            "video_analysis_id": video_analysis_id,
            "knowledge_version": load_knowledge(ROOT).package.version,
            "status": "complete",
            "cards": [
                {
                    "id": "claim-card-1",
                    "revision": 1,
                    "category": "claim",
                    "status": "conflicts",
                    "title": "与资料要求冲突",
                    "checked_statement": statement,
                    "fact_refs": ["claim-1"],
                    "action_ids": ["action-1"],
                    "evidence_ids": [evidence_id],
                    "explanation": "资料要求出现持续不适时停止。",
                    "suggested_action": "先停止并核对已有要求。",
                    "remaining_unknowns": [],
                    "handling_state": "open",
                }
            ],
            "context_items": [],
            "coverage_notes": ["只核查已送核候选"],
            "validation_notes": [],
        },
        "raw_responses": [{"text": "不应进入报告"}],
        "requests": [
            {
                "label": "具体说法资料对照",
                "elapsed_seconds": 5.0,
                "request_id": "request-claim",
                "response_model": "qwen-test",
                "usage": {"total_tokens": 50},
            }
        ],
    }


def preparation_run(*, video_analysis_id="video-report", scene_analysis_id="scene-report"):
    return {
        "result": {
            "schema_version": "0.1",
            "analysis_id": "preparation-report",
            "revision": 4,
            "video_analysis_id": video_analysis_id,
            "scene_analysis_id": scene_analysis_id,
            "knowledge_version": load_knowledge(ROOT).package.version,
            "status": "complete",
            "checked_rule_ids": ["R01", "R02", "R03", "R04", "R05", "R06"],
            "cards": [
                {
                    "id": "prep-card-1",
                    "revision": 4,
                    "category": "preparation",
                    "status": "condition_gap",
                    "title": "实际座椅有待调整的条件",
                    "rule_ids": ["R04"],
                    "fact_refs": ["action-1"],
                    "condition_refs": ["selected_equipment[eq-1].user_report.has_wheels"],
                    "evidence_ids": ["E04"],
                    "explanation": "本人确认实际座椅有轮。",
                    "suggested_action": "优先换用已有无轮座椅。",
                    "remaining_unknowns": ["防滑状态尚未确认"],
                    "handling_state": "user_reported_done",
                    "can_mark_done": True,
                }
            ],
            "coverage_notes": ["照片只提供候选观察"],
        },
        "input_snapshot": {
            "revision": 4,
            "area_name": "客厅",
            "user_conditions": {"current_discomfort": "no"},
            "selected_equipment": [{"id": "eq-1", "candidate_id": "chair-1"}],
        },
        "elapsed_seconds": 0.001,
        "model_requests": 0,
    }


def test_complete_report_combines_current_results_and_local_sources():
    run = build_report(
        video_run(),
        scene_run(),
        claim_run(),
        preparation_run(),
        load_knowledge(ROOT),
        revision=1,
    )
    report = run.result
    assert report.workflow_status == "complete"
    assert report.model_requests_during_report == 0
    assert {item.id for item in report.evidence} == {"E01", "E04"}
    assert report.status_counts == {"conflicts": 1, "condition_gap": 1}
    assert any("本人报告已处理" in item for item in report.unresolved_items)
    assert report.preparation_cards[0].status == "condition_gap"
    assert report.preparation_cards[0].handling_state == "user_reported_done"
    assert len(report.requests) == 2
    serialized = report_to_serializable(run)
    assert "raw_responses" not in serialized["result"]
    assert "不应进入报告" not in serialized["markdown"]
    assert "不是医疗诊断" in serialized["markdown"]


def test_missing_claim_and_preparation_produces_partial_not_pass():
    run = build_report(video_run(), None, None, None, load_knowledge(ROOT), revision=1)
    assert run.result.workflow_status == "partial"
    assert [item.status for item in run.result.stages] == ["complete", "not_started", "not_started"]
    assert "主张核查未形成当前结果" in run.result.unresolved_items
    assert "准备检查未形成当前结果" in run.result.unresolved_items
    assert "不能据此作整体判断" in run.result.conclusion


def test_partial_video_preserves_unprocessed_ranges():
    run = build_report(video_run(status="partial"), None, None, None, load_knowledge(ROOT), revision=1)
    assert run.result.workflow_status == "partial"
    assert any("20.0" in item and "25.0" in item for item in run.result.coverage_notes)


def test_stale_video_or_scene_result_is_rejected():
    knowledge = load_knowledge(ROOT)
    with pytest.raises(AnalysisError, match="旧视频"):
        build_report(video_run(), scene_run(), claim_run(video_analysis_id="video-old"), None, knowledge, revision=1)
    with pytest.raises(AnalysisError, match="旧场地"):
        build_report(
            video_run(),
            scene_run("scene-new"),
            claim_run(),
            preparation_run(scene_analysis_id="scene-old"),
            knowledge,
            revision=1,
        )


def test_invalid_evidence_id_is_rejected_instead_of_exported():
    with pytest.raises(AnalysisError, match="不存在的本地资料"):
        build_report(
            video_run(),
            scene_run(),
            claim_run(evidence_id="E99"),
            preparation_run(),
            load_knowledge(ROOT),
            revision=1,
        )


def test_outdated_knowledge_version_is_rejected():
    claims = claim_run()
    claims["result"]["knowledge_version"] = "old-version"
    with pytest.raises(AnalysisError, match="资料版本已过期"):
        build_report(video_run(), scene_run(), claims, None, load_knowledge(ROOT), revision=1)


def test_case_a_report_preserves_three_conflicts_and_one_insufficient_without_evidence():
    claims = claim_run()
    template = claims["result"]["cards"][0]
    claims["result"]["cards"] = [
        {**template, "id": "card-a2", "checked_statement": "不考虑身体差别", "evidence_ids": ["E03"]},
        {**template, "id": "card-a3", "checked_statement": "持续肿胀仍硬撑", "evidence_ids": ["E01"]},
        {
            **template,
            "id": "card-a4",
            "status": "insufficient",
            "title": "当前资料不足",
            "checked_statement": "练七天保证永不跌倒",
            "evidence_ids": [],
        },
        {**template, "id": "card-a5", "checked_statement": "第一天越多越好", "evidence_ids": ["E02"]},
    ]
    run = build_report(video_run(), None, claims, None, load_knowledge(ROOT), revision=1)
    assert run.result.status_counts == {"conflicts": 3, "insufficient": 1}
    insufficient = next(card for card in run.result.claim_cards if card.status == "insufficient")
    assert insufficient.evidence_ids == []
    assert {item.id for item in run.result.evidence} == {"E01", "E02", "E03"}


def test_case_c_report_keeps_insufficient_claim_and_rejected_context_separate():
    claims = claim_run(evidence_id="E01")
    card = claims["result"]["cards"][0]
    card.update(
        {
            "id": "card-c1",
            "status": "insufficient",
            "title": "当前资料不足",
            "checked_statement": "连续三天治好耳鸣",
            "evidence_ids": [],
        }
    )
    claims["result"]["context_items"] = [
        {
            "candidate_id": "candidate-c2",
            "checked_statement": "有人说越痛越有效，这种说法不要照做。",
            "fact_refs": ["claim-2"],
            "disposition": "context_only",
            "explanation": "视频明确反对该说法，不指控视频赞成。",
        }
    ]
    run = build_report(video_run(), None, claims, None, load_knowledge(ROOT), revision=1)
    assert run.result.claim_cards[0].status == "insufficient"
    assert run.result.claim_cards[0].evidence_ids == []
    assert run.result.context_items[0].disposition == "context_only"
    assert "不指控视频赞成" in run.markdown


def test_untrusted_model_text_is_escaped_in_markdown():
    run = build_report(
        video_run(),
        scene_run(),
        claim_run(statement="<img src=x>*错误* [链接]"),
        preparation_run(),
        load_knowledge(ROOT),
        revision=1,
    )
    assert "&lt;img src=x&gt;" in run.markdown
    assert "\\*错误\\*" in run.markdown
    assert "\\[链接\\]" in run.markdown


def test_failed_stage_is_recorded_without_inventing_results():
    run = build_report(
        video_run(),
        None,
        None,
        None,
        load_knowledge(ROOT),
        revision=1,
        claim_error="JSON结构错误",
        preparation_error="旧场地引用",
    )
    assert run.result.workflow_status == "partial"
    assert [item.status for item in run.result.stages] == ["complete", "failed", "failed"]
    assert not run.result.claim_cards
    assert not run.result.preparation_cards
