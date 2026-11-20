import pytest

from src.analysis import (
    AnalysisError,
    _mark_boundary_uncertain,
    _normalize_action_scope,
    _parse,
    _shift_time,
    _video_prompt,
)
from src.schemas import VideoSegmentExtraction


VALID = """{
  "actions": [],
  "claims": [{
    "start_seconds": 13,
    "end_seconds": 18,
    "text": "有人这样说，但视频不赞成",
    "text_kind": "paraphrase",
    "channel": "speech",
    "stance": "rejected",
    "recognition": "clear"
  }],
  "cautions": [],
  "uncertainties": []
}"""


def test_parses_fenced_json_and_preserves_rejected_stance():
    parsed, repaired = _parse(f"```json\n{VALID}\n```", VideoSegmentExtraction)
    assert parsed.claims[0].stance == "rejected"
    assert repaired is False


def test_limited_trailing_comma_repair():
    parsed, repaired = _parse(VALID.replace('"uncertainties": []', '"uncertainties": [],'), VideoSegmentExtraction)
    assert parsed.actions == []
    assert repaired is True


def test_missing_required_key_is_failure():
    with pytest.raises(AnalysisError, match="结构校验失败"):
        _parse(VALID.replace(',\n  "uncertainties": []', ""), VideoSegmentExtraction)


def test_unknown_time_must_be_a_pair():
    invalid = VALID.replace('"start_seconds": 13', '"start_seconds": null')
    with pytest.raises(AnalysisError, match="结构校验失败"):
        _parse(invalid, VideoSegmentExtraction)


def test_segment_time_maps_back_to_original_video():
    assert _shift_time(3.25, 20.0, 40.0) == 23.25
    assert _shift_time(25.0, 20.0, 40.0) == 40.0


def test_item_touching_split_boundary_becomes_uncertain():
    item = {"start_seconds": 0.2, "end_seconds": 2.0, "recognition": "clear"}
    assert _mark_boundary_uncertain(item, part_index=2, part_count=3, part_duration=15.0)
    assert item["recognition"] == "uncertain"


def test_video_prompt_keeps_speaker_stance_separate_from_truth_judgment():
    prompt = _video_prompt(30, has_audio=True, segmented=False)
    assert "直接主张、保证或建议某事用 endorsed" in prompt
    assert "即使该说法可能错误或危险" in prompt
    assert "不要照做/这种说法不对" in prompt
    assert "不能让字幕替代语音" in prompt
    assert "仍必须听取并提取这一时段实际语音" in prompt
    assert "不能根据语音要求反推画面器材属性" in prompt


def test_action_scope_is_deterministic_product_mapping():
    action = {"type": "sit_to_stand", "scope": "unsupported"}
    assert _normalize_action_scope(action) is True
    assert action["scope"] == "supported"
