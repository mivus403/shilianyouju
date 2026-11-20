from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import AnalysisError, RequestRecord, _parse
from .config import Settings
from .knowledge import EvidenceCard, KnowledgeBase
from .model_client import ModelAPIError, chat_completion_stream


ClaimStatus = Literal[
    "supported_limited",
    "conflicts",
    "overstated",
    "insufficient",
    "recognition_needed",
]
ClaimTopic = Literal[
    "persistent_discomfort",
    "unlimited_training_volume",
    "universal_suitability",
    "chair_requirements",
    "support_requirements",
    "general_fall_prevention",
    "floor_obstacles",
    "current_discomfort",
    "disease_effect",
    "other",
    "unclear",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimCandidate(StrictModel):
    candidate_id: str
    checked_statement: str = Field(min_length=1)
    fact_refs: list[str] = Field(min_length=1)
    source_kind: Literal["claim", "caution"]
    channel: Literal["speech", "caption"]
    stance: Literal["endorsed", "rejected", "quoted_only", "unclear"] | None
    recognition: Literal["clear", "uncertain", "unreadable"]
    start_seconds: float | None
    end_seconds: float | None
    action_ids: list[str]
    review_version: int


class ClaimAssessmentExtraction(StrictModel):
    candidate_id: str
    checked_statement: str = Field(min_length=1)
    fact_refs: list[str] = Field(min_length=1)
    action_ids: list[str]
    disposition: Literal["review", "context_only", "not_health_claim"]
    claim_scope: Literal["general_principle", "action_specific", "disease_effect", "equipment_condition", "unclear"]
    topics: list[ClaimTopic]
    status: ClaimStatus | None
    explanation: str = Field(min_length=1)
    evidence_ids: list[str]
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_disposition(self):
        if self.disposition == "review" and self.status is None:
            raise ValueError("review 项必须有 status")
        if self.disposition == "review" and not self.topics:
            raise ValueError("review 项必须有 topic；不能确认时填写 unclear")
        return self


class ClaimCheckExtraction(StrictModel):
    assessments: list[ClaimAssessmentExtraction]
    coverage_notes: list[str]


class ClaimCheckCard(StrictModel):
    id: str
    revision: int
    category: Literal["claim"] = "claim"
    status: ClaimStatus
    title: str
    checked_statement: str
    fact_refs: list[str]
    action_ids: list[str]
    evidence_ids: list[str]
    explanation: str
    suggested_action: str
    remaining_unknowns: list[str]
    handling_state: Literal["open"] = "open"


class ContextItem(StrictModel):
    candidate_id: str
    checked_statement: str
    fact_refs: list[str]
    disposition: Literal["context_only", "not_health_claim"]
    explanation: str


class ClaimCheckResult(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str
    revision: int
    video_analysis_id: str
    knowledge_version: str
    status: Literal["complete"] = "complete"
    cards: list[ClaimCheckCard]
    context_items: list[ContextItem]
    coverage_notes: list[str]
    validation_notes: list[str]


@dataclass(frozen=True)
class ClaimCheckRun:
    result: ClaimCheckResult
    candidates: list[ClaimCandidate]
    raw_response: dict[str, Any]
    request: RequestRecord


class ClaimCheckError(AnalysisError):
    def __init__(self, message: str, raw_response: dict[str, Any] | None = None, request: RequestRecord | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.request = request

    def details(self) -> dict[str, Any] | None:
        if not self.raw_response or not self.request:
            return None
        return {
            "raw_responses": [self.raw_response],
            "requests": [asdict(self.request)],
        }


def _infer_action_ids(fact: dict[str, Any], actions: list[dict[str, Any]]) -> list[str]:
    start = fact.get("start_seconds")
    end = fact.get("end_seconds")
    if start is None or end is None:
        return []
    matches = []
    for action in actions:
        if action.get("recognition") != "clear" or action.get("start_seconds") is None:
            continue
        if action["start_seconds"] <= end and action["end_seconds"] >= start:
            matches.append(action["id"])
    return matches if len(matches) == 1 else []


def build_claim_candidates(
    video_result: dict[str, Any], claim_reviews: dict[str, dict[str, Any]] | None = None
) -> list[ClaimCandidate]:
    reviews = claim_reviews or {}
    actions = video_result.get("actions", [])
    action_ids = {item["id"] for item in actions}
    facts = [("claim", item) for item in video_result.get("claims", [])]
    facts.extend(("caution", item) for item in video_result.get("cautions", []))
    seen: set[str] = set()
    candidates: list[ClaimCandidate] = []
    for source_kind, fact in facts:
        fact_id = fact["id"]
        if fact_id in seen:
            raise AnalysisError(f"视频事实 ID 重复：{fact_id}")
        seen.add(fact_id)
        review = reviews.get(fact_id, {})
        corrected = review.get("corrected_text")
        text = corrected.strip() if isinstance(corrected, str) and corrected.strip() else fact.get("corrected_text") or fact["text"]
        stance = review.get("stance", fact.get("stance"))
        if "action_ids" in review:
            linked_actions = list(review["action_ids"])
        else:
            linked_actions = _infer_action_ids(fact, actions)
        if not set(linked_actions) <= action_ids:
            raise AnalysisError(f"{fact_id} 的本地复核引用了不存在的动作")
        digest = hashlib.sha256(f"{fact_id}:{text}".encode("utf-8")).hexdigest()[:12]
        candidates.append(
            ClaimCandidate(
                candidate_id=f"candidate-{digest}",
                checked_statement=text,
                fact_refs=[fact_id],
                source_kind=source_kind,
                channel=fact["channel"],
                stance=stance,
                recognition="clear" if review.get("recognition_confirmed") is True else fact["recognition"],
                start_seconds=fact.get("start_seconds"),
                end_seconds=fact.get("end_seconds"),
                action_ids=linked_actions,
                review_version=int(review.get("version", 0)),
            )
        )
    return candidates


def _prompt(candidates: list[ClaimCandidate], knowledge: KnowledgeBase, actions: list[dict[str, Any]]) -> str:
    evidence = []
    for card in knowledge.package.cards:
        evidence.append(
            {
                "id": card.id,
                "topic": card.topic,
                "excerpt": card.excerpt,
                "context": card.context,
                "applicability": card.applicability,
                "exclusions": card.exclusions,
                "allowed_statement": card.allowed_statement,
                "match_scope": card.match_scope.model_dump(mode="json"),
            }
        )
    payload = {
        "actions": [
            {
                "id": item["id"],
                "type": item["type"],
                "description": item["description"],
                "recognition": item["recognition"],
                "scope": item["scope"],
            }
            for item in actions
        ],
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "evidence_cards": evidence,
    }
    statuses = knowledge.claim_policy["statuses"]
    return f"""你正在执行“视频具体说法与本地资料的对照”，不是给整段视频或个人下安全结论。只能使用下面提供的候选、动作和资料卡；不得使用外部知识，不得生成来源、URL、剂量、替代动作、诊断或治疗建议。

先保留完整原话、否定范围和说话者立场，再逐项核对人群、一般原则/具体动作、效果或条件、方向、时间剂量和前提。关键词只能帮助理解，不能直接决定状态。

状态含义：
{json.dumps(statuses, ensure_ascii=False)}

规则：
1. 每个 candidate_id 必须恰好返回一次；checked_statement、fact_refs、action_ids 必须原样复制。
2. 纯画面/文件说明用 not_health_claim。纯转述或完整反对语句若没有提出可单独核查的相反主张，用 context_only，不能指控视频赞成被反对的短语。
3. 原话、立场或动作关联不可靠时 disposition=review 且 status=recognition_needed。
4. 没有同一主张可用的资料时 status=insufficient 且 evidence_ids=[]；这不等于已证伪。
5. supported_limited、conflicts、overstated 必须引用适用的本地资料 ID。动作专属资料必须匹配关联动作。E06只适用于一般平衡/肌力/耐力锻炼的跌倒预防表述，不能证明某个具体动作有效。
6. 反对某说法的 stance=rejected 不等于视频赞成它，也不自动证明反对内容正确。caution 也可能需要核查，不能天然当作可信提醒。
7. explanation 只解释当前表述与资料适用范围的关系；不要输出纠正文案，应用会从本地资料卡读取允许表述。所有字符串保持简短，不在字符串内部使用引号或换行。
8. topics 只能从资料卡 match_scope.topics 或 disease_effect、other、unclear 中选择。review项至少一个主题；context_only或not_health_claim可用空数组。claim_scope 按主张本身填写，不按画面动作猜。

只返回 JSON，不要 Markdown，精确结构如下：
{{
  "assessments": [{{
    "candidate_id": "原样复制",
    "checked_statement": "原样复制完整表述",
    "fact_refs": ["原样复制"],
    "action_ids": ["原样复制"],
    "disposition": "review|context_only|not_health_claim",
    "claim_scope": "general_principle|action_specific|disease_effect|equipment_condition|unclear",
    "topics": ["资料主题或other/unclear"],
    "status": "supported_limited|conflicts|overstated|insufficient|recognition_needed|null",
    "explanation": "中文说明",
    "evidence_ids": ["E01"],
    "uncertainties": []
  }}],
  "coverage_notes": ["本次对照仍未覆盖的内容"]
}}

输入：
{json.dumps(payload, ensure_ascii=False)}"""


def _evidence_is_applicable(
    card: EvidenceCard,
    assessment: ClaimAssessmentExtraction,
    candidate: ClaimCandidate,
    actions: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    if not set(assessment.topics).intersection(card.match_scope.topics):
        return False, f"{card.id} 的主题与模型给出的主张主题不匹配"
    if card.match_scope.kind == "action_specific":
        linked_types = {actions[action_id]["type"] for action_id in candidate.action_ids if action_id in actions}
        if not linked_types.intersection(card.match_scope.actions):
            return False, f"{card.id} 是动作专属资料，但关联动作不匹配"
    if card.id == "E06" and assessment.claim_scope != "general_principle":
        return False, "E06 只支持一般跌倒预防表述，不能用于具体动作或疾病疗效"
    if card.id == "E06" and candidate.action_ids:
        general_categories = ("平衡", "肌力", "耐力", "锻炼类别", "一般锻炼")
        if not any(marker in candidate.checked_statement for marker in general_categories):
            return False, "E06 不能为已关联的具体动作提供跌倒预防效果依据"
    return True, None


def _coverage_notes(video_result: dict[str, Any]) -> list[str]:
    notes = list(video_result.get("uncertainties", []))
    if video_result.get("status") == "complete":
        notes.append("视频提取状态 complete 只表示请求和结构处理完整，不证明语音与画面内容全部识别。")
    if video_result.get("unprocessed_ranges"):
        notes.append(f"视频存在未处理区段：{video_result.get('unprocessed_ranges', [])}")
    speech = [
        item
        for group in (video_result.get("claims", []), video_result.get("cautions", []))
        for item in group
        if item.get("channel") == "speech"
    ]
    if video_result.get("audio_status") == "present" and not speech:
        notes.append("视频检测到音轨，但没有提取到语音说法；本次核查未覆盖语音，不能显示“视频没有说法”。")
    unsupported = [item["id"] for item in video_result.get("actions", []) if item.get("scope") == "unsupported"]
    if unsupported:
        notes.append("存在产品范围外动作：" + "、".join(unsupported) + "；只核查独立说法，不评价这些动作是否适练。")
    return list(dict.fromkeys(notes))


def _assemble_result(
    extraction: ClaimCheckExtraction,
    candidates: list[ClaimCandidate],
    video_result: dict[str, Any],
    knowledge: KnowledgeBase,
    revision: int,
) -> ClaimCheckResult:
    candidate_map = {item.candidate_id: item for item in candidates}
    returned_ids = [item.candidate_id for item in extraction.assessments]
    if len(returned_ids) != len(set(returned_ids)):
        raise AnalysisError("主张核查结果包含重复 candidate_id")
    if set(returned_ids) != set(candidate_map):
        missing = sorted(set(candidate_map) - set(returned_ids))
        extra = sorted(set(returned_ids) - set(candidate_map))
        raise AnalysisError(f"主张核查候选不完整；缺少 {missing}，多出 {extra}")

    actions = {item["id"]: item for item in video_result.get("actions", [])}
    cards: list[ClaimCheckCard] = []
    context_items: list[ContextItem] = []
    validation_notes: list[str] = []
    required_evidence = set(knowledge.claim_policy["required_evidence_statuses"])
    titles = {
        "supported_limited": "这项表述在有限范围内有依据",
        "conflicts": "这项表述与资料要求冲突",
        "overstated": "这项表述超出资料支持范围",
        "insufficient": "当前资料不足",
        "recognition_needed": "先确认原话或上下文",
    }

    for assessment in extraction.assessments:
        candidate = candidate_map[assessment.candidate_id]
        if assessment.checked_statement != candidate.checked_statement:
            raise AnalysisError(f"{candidate.candidate_id} 的 checked_statement 未原样保留")
        if set(assessment.fact_refs) != set(candidate.fact_refs):
            raise AnalysisError(f"{candidate.candidate_id} 的 fact_refs 与输入不一致")
        if set(assessment.action_ids) != set(candidate.action_ids):
            raise AnalysisError(f"{candidate.candidate_id} 的 action_ids 与输入不一致")
        unknown_ids = sorted(set(assessment.evidence_ids) - set(knowledge.cards))
        if unknown_ids:
            raise AnalysisError(f"模型引用了不存在的资料 ID：{unknown_ids}")

        conflicting_disposition = assessment.disposition != "review" and (
            assessment.status is not None or bool(assessment.evidence_ids)
        )
        needs_recognition = candidate.recognition != "clear" or candidate.stance == "unclear" or conflicting_disposition
        if assessment.disposition != "review" and not needs_recognition:
            context_items.append(
                ContextItem(
                    candidate_id=candidate.candidate_id,
                    checked_statement=candidate.checked_statement,
                    fact_refs=candidate.fact_refs,
                    disposition=assessment.disposition,
                    explanation=assessment.explanation,
                )
            )
            continue

        status = "recognition_needed" if needs_recognition else assessment.status
        assert status is not None
        uncertainties = list(assessment.uncertainties)
        if needs_recognition:
            if assessment.status != "recognition_needed":
                validation_notes.append(f"{candidate.candidate_id} 因识别、立场或核查分类尚未确认，状态改为 recognition_needed")
            status = "recognition_needed"
        if conflicting_disposition:
            note = "模型对该项的分类与结论互相矛盾，未采纳结论及引用，需要重新核查。"
            validation_notes.append(f"{candidate.candidate_id}：{note}")
            uncertainties.append(note)

        valid_evidence: list[str] = []
        if status not in {"insufficient", "recognition_needed"}:
            for evidence_id in assessment.evidence_ids:
                applicable, reason = _evidence_is_applicable(
                    knowledge.cards[evidence_id], assessment, candidate, actions
                )
                if applicable:
                    valid_evidence.append(evidence_id)
                elif reason:
                    validation_notes.append(f"{candidate.candidate_id}：已移除不适用引用；{reason}")
                    uncertainties.append(reason)
        elif assessment.evidence_ids:
            validation_notes.append(f"{candidate.candidate_id} 的 {status} 状态不采用资料引用")

        explanation = assessment.explanation
        if status in required_evidence and not valid_evidence:
            validation_notes.append(f"{candidate.candidate_id} 的确定性结论缺少适用资料，已降为 insufficient")
            status = "insufficient"
            explanation = knowledge.claim_policy["no_verdict_template"]
        if status in {"insufficient", "recognition_needed"}:
            valid_evidence = []

        if status == "recognition_needed":
            explanation = "原话、否定范围或说话者立场尚未可靠确认，本次不对这项说法作科学判断。"
            suggested_action = "先回看相关片段并更正原话、立场或动作关联，再重新核查。"
            if conflicting_disposition:
                explanation = "模型对该项的分类与结论互相矛盾，本次不采纳其科学判断或资料引用。"
                suggested_action = "回看原话，并重新进行资料核查；本次结果不能用作这项说法正确或错误的依据。"
        elif status == "insufficient":
            explanation = knowledge.claim_policy["no_verdict_template"]
            suggested_action = knowledge.claim_policy["no_verdict_template"]
        else:
            suggested_action = "；".join(
                dict.fromkeys(knowledge.cards[evidence_id].allowed_statement for evidence_id in valid_evidence)
            )

        cards.append(
            ClaimCheckCard(
                id=f"claim-card-{candidate.candidate_id.removeprefix('candidate-')}",
                revision=revision,
                status=status,
                title=titles[status],
                checked_statement=candidate.checked_statement,
                fact_refs=candidate.fact_refs,
                action_ids=candidate.action_ids,
                evidence_ids=valid_evidence,
                explanation=explanation,
                suggested_action=suggested_action,
                remaining_unknowns=list(dict.fromkeys(uncertainties)),
            )
        )

    if extraction.coverage_notes:
        validation_notes.append("模型生成的覆盖总结只保留在原始回复中，未作为系统覆盖结论。")
    return ClaimCheckResult(
        analysis_id=f"claim-check-{uuid.uuid4().hex[:12]}",
        revision=revision,
        video_analysis_id=video_result["analysis_id"],
        knowledge_version=knowledge.package.version,
        cards=cards,
        context_items=context_items,
        coverage_notes=_coverage_notes(video_result),
        validation_notes=validation_notes,
    )


def analyze_claims(
    video_result: dict[str, Any],
    claim_reviews: dict[str, dict[str, Any]],
    knowledge: KnowledgeBase,
    settings: Settings,
    *,
    revision: int,
    progress: Callable[[str], None] | None = None,
) -> ClaimCheckRun:
    candidates = build_claim_candidates(video_result, claim_reviews)
    if not candidates:
        raise AnalysisError("视频提取结果中没有说法或注意事项可供核查")
    if not settings.model_ready or not settings.api_key or not settings.base_url:
        raise AnalysisError(f"模型配置不完整：{', '.join(settings.missing_model_config)}")
    if progress:
        progress(f"正在用 {len(knowledge.cards)} 张本地资料卡对照 {len(candidates)} 个候选")
    started = time.perf_counter()
    try:
        response = chat_completion_stream(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            content=[{"type": "text", "text": _prompt(candidates, knowledge, video_result.get("actions", []))}],
            timeout_seconds=settings.request_timeout_seconds,
            reasoning_effort="low",
        )
    except ModelAPIError as exc:
        raise AnalysisError(str(exc)) from exc
    request = RequestRecord(
        "主张与资料对照",
        response.elapsed_seconds or time.perf_counter() - started,
        response.request_id,
        response.response_model,
        response.usage,
    )
    raw = {"label": "主张与资料对照", "json_repaired": None, "text": response.text}
    try:
        parsed, repaired = _parse(response.text, ClaimCheckExtraction)
        raw["json_repaired"] = repaired
        result = _assemble_result(parsed, candidates, video_result, knowledge, revision)
    except AnalysisError as exc:
        raw["parse_error"] = str(exc)
        raise ClaimCheckError(str(exc), raw, request) from exc
    return ClaimCheckRun(result=result, candidates=candidates, raw_response=raw, request=request)


def claim_check_to_serializable(run: ClaimCheckRun) -> dict[str, Any]:
    return {
        "result": run.result.model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in run.candidates],
        "raw_responses": [run.raw_response],
        "requests": [asdict(run.request)],
    }
