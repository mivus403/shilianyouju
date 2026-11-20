from copy import deepcopy
from pathlib import Path

import pytest

from src.analysis import AnalysisError
from src.claim_check import (
    ClaimAssessmentExtraction,
    ClaimCheckExtraction,
    _assemble_result,
    _prompt,
    build_claim_candidates,
)
from src.knowledge import load_knowledge


ROOT = Path(__file__).resolve().parents[1]


def sample_video_result():
    return {
        "analysis_id": "video-test",
        "revision": 1,
        "audio_status": "present",
        "status": "complete",
        "unprocessed_ranges": [],
        "actions": [
            {
                "id": "action-1",
                "start_seconds": 0.0,
                "end_seconds": 35.0,
                "type": "seated_arm_movement",
                "description": "坐姿手臂活动",
                "equipment": ["chair"],
                "recognition": "clear",
                "scope": "supported",
            }
        ],
        "claims": [
            {
                "id": "claim-1",
                "start_seconds": 5.0,
                "end_seconds": 11.0,
                "text": "所有老人都能照做，不必考虑身体差别。",
                "corrected_text": None,
                "text_kind": "quote",
                "channel": "speech",
                "stance": "endorsed",
                "recognition": "clear",
            },
            {
                "id": "claim-2",
                "start_seconds": 21.0,
                "end_seconds": 26.0,
                "text": "这套动作保证七天后永不跌倒。",
                "corrected_text": None,
                "text_kind": "quote",
                "channel": "speech",
                "stance": "endorsed",
                "recognition": "clear",
            },
            {
                "id": "claim-3",
                "start_seconds": 27.0,
                "end_seconds": 31.0,
                "text": "有人说越痛越有效，这种说法不要照做。",
                "corrected_text": None,
                "text_kind": "quote",
                "channel": "speech",
                "stance": "rejected",
                "recognition": "clear",
            },
        ],
        "cautions": [
            {
                "id": "caution-1",
                "start_seconds": 12.0,
                "end_seconds": 20.0,
                "text": "持续肿胀且休息后未缓解也要继续硬撑。",
                "text_kind": "quote",
                "channel": "caption",
                "recognition": "clear",
            }
        ],
        "uncertainties": [],
    }


def assessment(candidate, *, disposition="review", status="insufficient", scope="action_specific", topics=None, evidence=None):
    return ClaimAssessmentExtraction(
        candidate_id=candidate.candidate_id,
        checked_statement=candidate.checked_statement,
        fact_refs=candidate.fact_refs,
        action_ids=candidate.action_ids,
        disposition=disposition,
        claim_scope=scope,
        topics=topics or ["other"],
        status=status if disposition == "review" else None,
        explanation="按完整表述和资料适用范围核对。",
        evidence_ids=evidence or [],
        uncertainties=[],
    )


def test_loads_checked_local_knowledge_and_prompt_has_no_source_urls():
    knowledge = load_knowledge(ROOT)
    assert knowledge.package.version == "2026-09-22.3"
    assert len(knowledge.cards) == 8
    candidates = build_claim_candidates(sample_video_result())
    prompt = _prompt(candidates, knowledge, sample_video_result()["actions"])
    assert "https://" not in prompt
    assert "expected.json" not in prompt
    assert "E01" in prompt and "allowed_statement" in prompt


def test_local_correction_does_not_overwrite_original_fact():
    video = sample_video_result()
    original = deepcopy(video)
    candidates = build_claim_candidates(
        video,
        {
            "claim-1": {
                "corrected_text": "所有老人都能照做，而且不必考虑身体差别。",
                "stance": "endorsed",
                "action_ids": ["action-1"],
                "version": 2,
            }
        },
    )
    corrected = next(item for item in candidates if item.fact_refs == ["claim-1"])
    assert corrected.checked_statement.startswith("所有老人")
    assert corrected.review_version == 2
    assert video == original


def test_cautions_are_checked_and_semantically_invalid_evidence_is_downgraded():
    video = sample_video_result()
    knowledge = load_knowledge(ROOT)
    candidates = build_claim_candidates(video)
    by_ref = {item.fact_refs[0]: item for item in candidates}
    extraction = ClaimCheckExtraction(
        assessments=[
            assessment(
                by_ref["claim-1"],
                status="conflicts",
                scope="general_principle",
                topics=["universal_suitability"],
                evidence=["E03"],
            ),
            assessment(
                by_ref["claim-2"],
                status="overstated",
                scope="general_principle",
                topics=["general_fall_prevention"],
                evidence=["E06"],
            ),
            assessment(by_ref["claim-3"], disposition="context_only", status=None),
            assessment(
                by_ref["caution-1"],
                status="conflicts",
                scope="general_principle",
                topics=["persistent_discomfort"],
                evidence=["E01"],
            ),
        ],
        coverage_notes=[],
    )
    result = _assemble_result(extraction, candidates, video, knowledge, revision=1)
    cards = {card.fact_refs[0]: card for card in result.cards}
    assert cards["claim-1"].status == "conflicts"
    assert cards["claim-1"].evidence_ids == ["E03"]
    assert cards["caution-1"].status == "conflicts"
    assert cards["caution-1"].evidence_ids == ["E01"]
    assert cards["claim-2"].status == "insufficient"
    assert cards["claim-2"].evidence_ids == []
    assert any("E06" in note for note in result.validation_notes)
    assert result.context_items[0].fact_refs == ["claim-3"]


def test_nonexistent_evidence_id_fails_the_whole_model_result():
    video = sample_video_result()
    knowledge = load_knowledge(ROOT)
    candidates = build_claim_candidates(video)
    assessments = []
    for candidate in candidates:
        if candidate.fact_refs == ["claim-1"]:
            assessments.append(
                assessment(
                    candidate,
                    status="conflicts",
                    scope="general_principle",
                    topics=["universal_suitability"],
                    evidence=["E99"],
                )
            )
        else:
            assessments.append(assessment(candidate))
    extraction = ClaimCheckExtraction(assessments=assessments, coverage_notes=[])
    with pytest.raises(AnalysisError, match="不存在的资料 ID"):
        _assemble_result(extraction, candidates, video, knowledge, revision=1)


def test_e06_can_support_only_an_explicit_general_exercise_category_claim():
    video = sample_video_result()
    video["claims"][1]["text"] = "平衡、肌力和耐力锻炼保证以后永不跌倒。"
    knowledge = load_knowledge(ROOT)
    candidates = build_claim_candidates(video)
    assessments = []
    for candidate in candidates:
        if candidate.fact_refs == ["claim-2"]:
            assessments.append(
                assessment(
                    candidate,
                    status="overstated",
                    scope="general_principle",
                    topics=["general_fall_prevention"],
                    evidence=["E06"],
                )
            )
        else:
            assessments.append(assessment(candidate, disposition="context_only", status=None))
    result = _assemble_result(
        ClaimCheckExtraction(assessments=assessments, coverage_notes=[]),
        candidates,
        video,
        knowledge,
        revision=1,
    )
    assert result.cards[0].status == "overstated"
    assert result.cards[0].evidence_ids == ["E06"]
