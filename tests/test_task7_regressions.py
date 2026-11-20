"""结论边界回归测试，使用明确标注的故障注入。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analysis import analyze_scene
from src.claim_check import ClaimAssessmentExtraction, ClaimCheckExtraction, _assemble_result, build_claim_candidates
from src.knowledge import load_knowledge
from src.media import validate_image
from src.model_client import ModelResponse
from src.preparation import evaluate_preparation
from test_claim_check import assessment, sample_video_result
from test_preparation import preparation, scene, video

ROOT = Path(__file__).resolve().parents[1]


def test_uncertain_fact_cannot_bypass_gate_as_context():
    source = sample_video_result()
    source["claims"][0]["recognition"] = "uncertain"
    candidates = build_claim_candidates(source)
    parsed = ClaimCheckExtraction(
        assessments=[assessment(c, disposition="context_only") for c in candidates], coverage_notes=[]
    )
    result = _assemble_result(parsed, candidates, source, load_knowledge(ROOT), 1)
    card = next(c for c in result.cards if c.fact_refs == ["claim-1"])
    assert card.status == "recognition_needed"
    assert not card.evidence_ids


def test_uncertain_gate_does_not_retain_an_assertive_model_explanation():
    source = sample_video_result()
    source["claims"][0]["recognition"] = "uncertain"
    candidates = build_claim_candidates(source)
    rows = [assessment(c, status="conflicts", topics=["universal_suitability"], evidence=["E03"]) for c in candidates]
    rows[0].explanation = "已经确认原话错误，所有人都不应练。"
    result = _assemble_result(ClaimCheckExtraction(assessments=rows, coverage_notes=[]), candidates, source, load_knowledge(ROOT), 1)
    assert result.cards[0].status == "recognition_needed"
    assert "已经确认" not in result.cards[0].explanation


def test_explicit_replay_confirmation_can_resolve_recognition_without_mutating_source():
    source = sample_video_result()
    source["claims"][0]["recognition"] = "uncertain"
    original = deepcopy(source)
    review = {"claim-1": {"corrected_text": "运动应结合个人身体条件。", "stance": "endorsed", "recognition_confirmed": True, "version": 1}}
    assert build_claim_candidates(source, review)[0].recognition == "clear"
    review["claim-1"]["recognition_confirmed"] = False
    assert build_claim_candidates(source, review)[0].recognition == "uncertain"
    assert source == original


@pytest.mark.parametrize("evidence_id,text,topic", [
    ("E01", "关节持续肿胀且休息后未缓解，应先暂停。", "persistent_discomfort"),
    ("E03", "运动要结合个人身体情况选择。", "universal_suitability"),
    ("E08", "身体不适时，不要勉强跟练。", "current_discomfort"),
])
def test_supported_advice_does_not_inherit_accusation_or_invent_user_symptoms(evidence_id, text, topic):
    source = sample_video_result()
    source["claims"] = [{**source["claims"][0], "text": text}]
    source["cautions"] = []
    candidates = build_claim_candidates(source)
    row = assessment(candidates[0], status="supported_limited", scope="general_principle", topics=[topic], evidence=[evidence_id])
    result = _assemble_result(ClaimCheckExtraction(assessments=[row], coverage_notes=[]), candidates, source, load_knowledge(ROOT), 1)
    correction = result.cards[0].suggested_action
    assert "视频要求" not in correction
    assert "视频中无需考虑" not in correction
    assert "您填写" not in correction


def test_declared_restriction_without_details_remains_unresolved():
    room = scene()
    setup = preparation(scene_result=room)
    setup.user_conditions.restriction_status = "has_restrictions"
    setup.user_conditions.restrictions = []
    setup.user_conditions.restriction_note = None
    result = evaluate_preparation(video(), room, setup, load_knowledge(ROOT), revision=1).result
    cards = [card for card in result.cards if "R02" in card.rule_ids]
    assert len(cards) == 1
    assert cards[0].status == "needs_confirmation"


def test_no_medical_conclusion_can_be_added_by_a_user_correction():
    source = sample_video_result()
    review = {"claim-1": {"corrected_text": "运动应结合个人身体条件。", "version": 1}}
    source["claims"][0]["recognition"] = "uncertain"
    assert build_claim_candidates(source, review)[0].recognition == "uncertain"


def test_empty_topics_are_allowed_only_for_non_verdict_items():
    candidate = build_claim_candidates(sample_video_result())[0]
    data = assessment(candidate, disposition="not_health_claim").model_dump()
    data["topics"] = []
    assert ClaimAssessmentExtraction.model_validate(data).disposition == "not_health_claim"
    data.update(disposition="review", status="supported_limited")
    with pytest.raises(ValidationError):
        ClaimAssessmentExtraction.model_validate(data)


def test_conflicting_model_classification_becomes_unresolved_not_a_verdict():
    source = sample_video_result()
    candidates = build_claim_candidates(source)
    rows = [assessment(c, disposition="context_only") for c in candidates]
    raw = rows[0].model_dump()
    raw.update(status="supported_limited", evidence_ids=["E03"], explanation="该视频完全正确。")
    rows[0] = ClaimAssessmentExtraction.model_validate(raw)
    result = _assemble_result(ClaimCheckExtraction(assessments=rows, coverage_notes=[]), candidates, source, load_knowledge(ROOT), 1)
    assert result.cards[0].status == "recognition_needed"
    assert result.cards[0].evidence_ids == []
    assert "互相矛盾" in result.cards[0].explanation
    assert "完全正确" not in result.cards[0].explanation


@pytest.mark.parametrize("attribute_name", ["is_stable", "stability_confirmed", "non_slip_confirmed", "load_capacity"])
def test_photo_cannot_certify_physical_properties_under_aliases(monkeypatch, attribute_name):
    raw = {"visible_items": [], "equipment_candidates": [{"candidate_id": "chair-1", "type": "chair", "photo_ids": ["photo-1"], "description": "椅子", "recognition": "clear", "attributes": [{"name": attribute_name, "value": "present", "evidence": "只依据照片", "recognition": "clear"}]}], "observations": [], "uncertainties": []}
    monkeypatch.setattr("src.analysis._call", lambda *_: ModelResponse(json.dumps(raw), 0.01, "injected", "test", None))
    image = validate_image(ROOT / "samples/case_b_preparation/scene_before_overview.png")
    result = analyze_scene([image], None, revision=1).result
    assert result.status == "failed"
    assert not result.equipment_candidates


@pytest.mark.parametrize("audio_status,expected", [("present", "partial"), ("absent", "complete")])
def test_missing_speech_with_audio_cannot_be_complete(monkeypatch, audio_status, expected):
    from src.analysis import analyze_video
    from src.media import MediaPart, PreparedVideo, VideoMetadata

    raw = '{"actions": [], "claims": [], "cautions": [], "uncertainties": []}'
    monkeypatch.setattr("src.analysis._call", lambda *_: ModelResponse(raw, 0.01, None, "test", None))
    path = ROOT / "samples/task7/I.mp4"
    prepared = PreparedVideo("direct", [MediaPart(path, 0, 9)], path.stat().st_size, path.stat().st_size, 0)
    metadata = VideoMetadata(9, path.stat().st_size, "mp4", "h264", 960, 540, audio_status, None, 0)
    result = analyze_video(prepared, metadata, None, revision=1, video_id="missing-speech-test").result
    assert result.status == expected
    if audio_status == "present":
        assert any("未提取任何语音项" in note for note in result.uncertainties)


@pytest.mark.parametrize("name,value", [("base_type", "cantilever"), ("has_wheels", "no"), ("stability", "safe")])
def test_non_contract_photo_attribute_is_unknown_not_a_confirmed_fact(name, value):
    from src.schemas import CandidateAttribute

    result = CandidateAttribute.model_validate({"name": name, "value": value, "evidence": "模型声称已确定", "recognition": "clear"})
    assert result.value == "unknown"
    assert result.recognition == "uncertain"
    assert "模型声称已确定" not in result.evidence


def test_other_malformed_photo_fields_are_still_rejected():
    from src.schemas import CandidateAttribute

    with pytest.raises(ValidationError):
        CandidateAttribute.model_validate({"name": "base_type", "value": "cantilever", "unexpected": "不能吞掉其他结构错误"})


@pytest.mark.parametrize("description", ["木椅，结构稳固但尺寸较小", "座面有一定硬度", "台面高度适中", "稳固的四脚支撑结构"])
def test_free_text_cannot_bypass_photo_physical_property_boundary(monkeypatch, description):
    raw = {"visible_items": [], "equipment_candidates": [{"candidate_id": "chair-1", "type": "chair", "photo_ids": ["photo-1"], "description": description, "attributes": [], "recognition": "clear"}], "observations": [], "uncertainties": []}
    monkeypatch.setattr("src.analysis._call", lambda *_: ModelResponse(json.dumps(raw), 0, None, "test", None))
    photo = validate_image(ROOT / "samples/case_b_preparation/scene_before_overview.png")
    result = analyze_scene([photo], None, revision=1).result
    assert result.status == "complete"
    candidate = result.equipment_candidates[0]
    assert candidate.recognition == "uncertain"
    assert "不能仅凭照片确认" in candidate.description
    assert description not in candidate.description
    assert result.uncertainties


def test_uncertain_wheels_cannot_be_displayed_as_a_definite_photo_attribute(monkeypatch):
    raw = {"visible_items": [], "equipment_candidates": [{"candidate_id": "chair-1", "type": "chair", "photo_ids": ["photo-1"], "description": "底部被遮挡的椅子", "attributes": [{"name": "has_wheels", "value": "present", "evidence": "隐约可见滚轮结构", "recognition": "uncertain"}], "recognition": "uncertain"}], "observations": [], "uncertainties": []}
    monkeypatch.setattr("src.analysis._call", lambda *_: ModelResponse(json.dumps(raw), 0, None, "test", None))
    photo = validate_image(ROOT / "samples/case_b_preparation/scene_before_overview.png")
    run = analyze_scene([photo], None, revision=1)
    assert run.result.equipment_candidates[0].attributes[0].value == "unknown"
    assert run.raw_responses[0]["normalizations"]
