from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Recognition = Literal["clear", "uncertain", "unreadable"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimedModel(StrictModel):
    start_seconds: float | None
    end_seconds: float | None

    @model_validator(mode="after")
    def validate_time_pair(self):
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("start_seconds 和 end_seconds 必须同时为 null 或同时有值")
        if self.start_seconds is not None:
            if self.start_seconds < 0:
                raise ValueError("start_seconds 不能为负数")
            if self.end_seconds < self.start_seconds:
                raise ValueError("end_seconds 不能早于 start_seconds")
        return self


class ActionExtraction(TimedModel):
    type: Literal[
        "seated_arm_movement",
        "sit_to_stand",
        "supported_calf_raise",
        "other",
        "unknown",
    ]
    description: str = Field(min_length=1)
    equipment: list[str]
    recognition: Recognition
    scope: Literal["supported", "unsupported", "unknown"]


class ClaimExtraction(TimedModel):
    text: str = Field(min_length=1)
    text_kind: Literal["quote", "paraphrase"]
    channel: Literal["speech", "caption"]
    stance: Literal["endorsed", "rejected", "quoted_only", "unclear"]
    recognition: Recognition


class CautionExtraction(TimedModel):
    text: str = Field(min_length=1)
    text_kind: Literal["quote", "paraphrase"]
    channel: Literal["speech", "caption"]
    recognition: Recognition


class VideoSegmentExtraction(StrictModel):
    actions: list[ActionExtraction]
    claims: list[ClaimExtraction]
    cautions: list[CautionExtraction]
    uncertainties: list[str]


class VideoAction(ActionExtraction):
    id: str


class VideoClaim(ClaimExtraction):
    id: str
    corrected_text: str | None = None


class VideoCaution(CautionExtraction):
    id: str


class VideoAnalysis(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str
    revision: int = Field(ge=1)
    video_id: str
    duration_seconds: float = Field(gt=0)
    audio_status: Literal["present", "absent", "unreadable"]
    status: Literal["complete", "partial", "failed"]
    unprocessed_ranges: list[list[float]]
    actions: list[VideoAction]
    claims: list[VideoClaim]
    cautions: list[VideoCaution]
    uncertainties: list[str]


class VisibleItemExtraction(StrictModel):
    photo_id: str
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    recognition: Recognition


class CandidateAttribute(StrictModel):
    name: str = Field(min_length=1)
    value: Literal["present", "absent", "unknown"]
    evidence: str = Field(min_length=1)
    recognition: Recognition

    @model_validator(mode="before")
    @classmethod
    def keep_unrecognized_values_unknown(cls, data):
        if isinstance(data, dict) and all(key in data for key in ("name", "value", "evidence", "recognition")) and data["value"] not in ("present", "absent", "unknown"):
            return {
                **data,
                "value": "unknown",
                "recognition": "uncertain",
                "evidence": "模型返回的属性值不在约定范围，本项保持未知；可查看原始回复。",
            }
        return data


class EquipmentCandidateExtraction(StrictModel):
    candidate_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    photo_ids: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    attributes: list[CandidateAttribute]
    recognition: Recognition


class SceneObservationExtraction(StrictModel):
    photo_id: str
    equipment_candidate_id: str | None
    kind: Literal[
        "visible_wheels",
        "possible_obstacle",
        "support_visible",
        "view_insufficient",
        "other",
    ]
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)
    recognition: Recognition


class SceneExtraction(StrictModel):
    visible_items: list[VisibleItemExtraction]
    equipment_candidates: list[EquipmentCandidateExtraction]
    observations: list[SceneObservationExtraction]
    uncertainties: list[str]


class ScenePhoto(StrictModel):
    id: str
    role: Literal["overview", "floor", "equipment", "extra"] = "extra"
    path: str


class SceneAnalysis(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str
    revision: int = Field(ge=1)
    status: Literal["complete", "failed"]
    photos: list[ScenePhoto]
    selected_equipment: list = Field(default_factory=list)
    visible_items: list[VisibleItemExtraction]
    equipment_candidates: list[EquipmentCandidateExtraction]
    observations: list[SceneObservationExtraction]
    uncertainties: list[str]
