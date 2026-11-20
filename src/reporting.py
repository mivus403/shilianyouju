from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .analysis import AnalysisError
from .claim_check import ClaimCheckResult
from .knowledge import KnowledgeBase
from .preparation import PreparationResult
from .schemas import SceneAnalysis, VideoAnalysis


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportStage(StrictModel):
    name: Literal["video", "claim", "preparation"]
    status: Literal["not_started", "complete", "partial", "failed"]
    analysis_id: str | None
    revision: int | None
    note: str


class ReportAction(StrictModel):
    id: str
    type: str
    description: str
    start_seconds: float | None
    end_seconds: float | None
    recognition: str
    scope: str


class ReportClaimCard(StrictModel):
    id: str
    status: str
    title: str
    checked_statement: str
    fact_refs: list[str]
    action_ids: list[str]
    evidence_ids: list[str]
    explanation: str
    suggested_action: str
    remaining_unknowns: list[str]


class ReportContextItem(StrictModel):
    candidate_id: str
    checked_statement: str
    fact_refs: list[str]
    disposition: str
    explanation: str


class ReportPreparationCard(StrictModel):
    id: str
    status: str
    title: str
    rule_ids: list[str]
    fact_refs: list[str]
    condition_refs: list[str]
    evidence_ids: list[str]
    explanation: str
    suggested_action: str
    remaining_unknowns: list[str]
    handling_state: str


class ReportEvidence(StrictModel):
    id: str
    source_title: str
    publisher: str
    url: str
    excerpt: str
    applicability: str
    exclusions: list[str]
    checked_at: str
    version: str


class ReportRequest(StrictModel):
    stage: Literal["video", "scene", "claim"]
    label: str
    elapsed_seconds: float
    request_id: str | None
    response_model: str | None
    usage: dict[str, Any] | None


class FormalReport(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    report_id: str
    revision: int = Field(ge=1)
    generated_at: str
    title: str
    workflow_status: Literal["complete", "partial", "failed"]
    boundary_statement: str
    conclusion: str
    knowledge_version: str
    input_versions: dict[str, Any]
    stages: list[ReportStage]
    actions: list[ReportAction]
    claim_cards: list[ReportClaimCard]
    context_items: list[ReportContextItem]
    preparation_cards: list[ReportPreparationCard]
    preparation_input: dict[str, Any] | None
    status_counts: dict[str, int]
    coverage_notes: list[str]
    unresolved_items: list[str]
    next_actions: list[str]
    evidence: list[ReportEvidence]
    requests: list[ReportRequest]
    model_requests_during_report: Literal[0] = 0


@dataclass(frozen=True)
class ReportRun:
    result: FormalReport
    markdown: str
    elapsed_seconds: float


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _request_records(stage: str, run_data: dict[str, Any] | None) -> list[ReportRequest]:
    if not run_data:
        return []
    return [
        ReportRequest(
            stage=stage,
            label=item["label"],
            elapsed_seconds=item["elapsed_seconds"],
            request_id=item.get("request_id"),
            response_model=item.get("response_model"),
            usage=item.get("usage"),
        )
        for item in run_data.get("requests", [])
    ]


def _validate_versions(
    video: VideoAnalysis,
    scene: SceneAnalysis | None,
    claim: ClaimCheckResult | None,
    preparation: PreparationResult | None,
    knowledge: KnowledgeBase,
) -> None:
    if claim:
        if claim.video_analysis_id != video.analysis_id:
            raise AnalysisError("主张核查结果属于旧视频，不能生成当前报告")
        if claim.knowledge_version != knowledge.package.version:
            raise AnalysisError("主张核查使用的资料版本已过期，不能生成当前报告")
    if preparation:
        if preparation.video_analysis_id != video.analysis_id:
            raise AnalysisError("准备检查结果属于旧视频，不能生成当前报告")
        if preparation.knowledge_version != knowledge.package.version:
            raise AnalysisError("准备检查使用的资料版本已过期，不能生成当前报告")
        current_scene_id = scene.analysis_id if scene else None
        if preparation.scene_analysis_id != current_scene_id:
            raise AnalysisError("准备检查结果属于旧场地或未包含当前场地，不能生成当前报告")


def _stage_records(
    video: VideoAnalysis,
    claim: ClaimCheckResult | None,
    preparation: PreparationResult | None,
    claim_error: str | None,
    preparation_error: str | None,
) -> list[ReportStage]:
    video_note = {
        "complete": "请求和结构处理完整，不代表所有声音与画面均已识别。",
        "partial": "仅使用成功处理的范围，未处理时间必须保留。",
        "failed": "视频提取失败，后续动作相关结论不能完成。",
    }[video.status]
    if claim:
        claim_stage = ReportStage(
            name="claim",
            status=claim.status,
            analysis_id=claim.analysis_id,
            revision=claim.revision,
            note="已对已送核候选完成本地资料对照；不代表视频全部说法均被识别。",
        )
    elif claim_error:
        claim_stage = ReportStage(
            name="claim",
            status="failed",
            analysis_id=None,
            revision=None,
            note="主张核查失败：" + claim_error,
        )
    else:
        claim_stage = ReportStage(
            name="claim",
            status="not_started",
            analysis_id=None,
            revision=None,
            note="尚未形成主张核查结果。",
        )
    if preparation:
        preparation_stage = ReportStage(
            name="preparation",
            status=preparation.status,
            analysis_id=preparation.analysis_id,
            revision=preparation.revision,
            note="准备状态只表示规则覆盖程度，不是个人适练或场地安全结论。",
        )
    elif preparation_error:
        preparation_stage = ReportStage(
            name="preparation",
            status="failed",
            analysis_id=None,
            revision=None,
            note="准备检查失败：" + preparation_error,
        )
    else:
        preparation_stage = ReportStage(
            name="preparation",
            status="not_started",
            analysis_id=None,
            revision=None,
            note="尚未提交本人条件与准备检查。",
        )
    return [
        ReportStage(
            name="video",
            status=video.status,
            analysis_id=video.analysis_id,
            revision=video.revision,
            note=video_note,
        ),
        claim_stage,
        preparation_stage,
    ]


def _workflow_status(stages: list[ReportStage]) -> str:
    video = stages[0]
    if video.status == "failed":
        return "failed"
    return "complete" if all(stage.status == "complete" for stage in stages) else "partial"


def _status_counts(
    claim_cards: list[ReportClaimCard],
    preparation_cards: list[ReportPreparationCard],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in [*claim_cards, *preparation_cards]:
        counts[card.status] = counts.get(card.status, 0) + 1
    return counts


def _report_id(input_versions: dict[str, Any], revision: int) -> str:
    payload = json.dumps(
        {"input_versions": input_versions, "report_revision": revision},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return "report-" + hashlib.sha256(payload).hexdigest()[:12]


def build_report(
    video_run: dict[str, Any],
    scene_run: dict[str, Any] | None,
    claim_run: dict[str, Any] | None,
    preparation_run: dict[str, Any] | None,
    knowledge: KnowledgeBase,
    *,
    revision: int,
    claim_error: str | None = None,
    preparation_error: str | None = None,
) -> ReportRun:
    started = time.perf_counter()
    try:
        video = VideoAnalysis.model_validate(video_run["result"])
        scene = SceneAnalysis.model_validate(scene_run["result"]) if scene_run else None
        claim = ClaimCheckResult.model_validate(claim_run["result"]) if claim_run else None
        preparation = PreparationResult.model_validate(preparation_run["result"]) if preparation_run else None
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"报告输入结构无效：{exc}") from exc
    _validate_versions(video, scene, claim, preparation, knowledge)

    stages = _stage_records(video, claim, preparation, claim_error, preparation_error)
    workflow_status = _workflow_status(stages)
    input_versions = {
        "video": {"analysis_id": video.analysis_id, "revision": video.revision},
        "scene": {"analysis_id": scene.analysis_id, "revision": scene.revision} if scene else None,
        "claim": {"analysis_id": claim.analysis_id, "revision": claim.revision} if claim else None,
        "preparation": {"analysis_id": preparation.analysis_id, "revision": preparation.revision} if preparation else None,
        "knowledge": knowledge.package.version,
    }
    claim_cards = [
        ReportClaimCard(
            id=card.id,
            status=card.status,
            title=card.title,
            checked_statement=card.checked_statement,
            fact_refs=card.fact_refs,
            action_ids=card.action_ids,
            evidence_ids=card.evidence_ids,
            explanation=card.explanation,
            suggested_action=card.suggested_action,
            remaining_unknowns=card.remaining_unknowns,
        )
        for card in (claim.cards if claim else [])
    ]
    context_items = [ReportContextItem(**item.model_dump()) for item in (claim.context_items if claim else [])]
    preparation_cards = [
        ReportPreparationCard(
            id=card.id,
            status=card.status,
            title=card.title,
            rule_ids=card.rule_ids,
            fact_refs=card.fact_refs,
            condition_refs=card.condition_refs,
            evidence_ids=card.evidence_ids,
            explanation=card.explanation,
            suggested_action=card.suggested_action,
            remaining_unknowns=card.remaining_unknowns,
            handling_state=card.handling_state,
        )
        for card in (preparation.cards if preparation else [])
    ]
    evidence_ids = _unique(
        [evidence_id for card in claim_cards for evidence_id in card.evidence_ids]
        + [evidence_id for card in preparation_cards for evidence_id in card.evidence_ids]
    )
    invalid_evidence = [evidence_id for evidence_id in evidence_ids if evidence_id not in knowledge.cards]
    if invalid_evidence:
        raise AnalysisError("报告包含不存在的本地资料 ID：" + "、".join(invalid_evidence))
    evidence = [
        ReportEvidence(
            id=evidence_id,
            source_title=knowledge.cards[evidence_id].source_title,
            publisher=knowledge.cards[evidence_id].publisher,
            url=str(knowledge.cards[evidence_id].url),
            excerpt=knowledge.cards[evidence_id].excerpt,
            applicability=knowledge.cards[evidence_id].applicability,
            exclusions=knowledge.cards[evidence_id].exclusions,
            checked_at=knowledge.cards[evidence_id].checked_at,
            version=knowledge.cards[evidence_id].version,
        )
        for evidence_id in evidence_ids
    ]

    coverage_notes: list[str] = []
    if video.unprocessed_ranges:
        coverage_notes.append(f"视频未处理时间范围：{video.unprocessed_ranges}")
    coverage_notes.extend(video.uncertainties)
    if claim:
        coverage_notes.extend(claim.coverage_notes)
        coverage_notes.extend(claim.validation_notes)
    elif claim_error:
        coverage_notes.append("主张核查失败，报告没有主张结论。")
    else:
        coverage_notes.append("主张核查尚未完成，报告没有主张结论。")
    if preparation:
        coverage_notes.extend(preparation.coverage_notes)
    elif preparation_error:
        coverage_notes.append("准备检查失败，报告没有本人条件与场地规则结论。")
    else:
        coverage_notes.append("准备检查尚未完成，报告没有本人条件与场地规则结论。")
    if scene is None:
        coverage_notes.append("尚无场地图片提取结果。")
    elif scene.status == "failed":
        coverage_notes.append("场地图片提取失败，不能据此确认器材或活动位置。")
    else:
        coverage_notes.extend(scene.uncertainties)

    unresolved_items: list[str] = []
    for card in claim_cards:
        if card.status != "supported_limited":
            unresolved_items.append(f"主张 {card.id}：{card.title}——{card.checked_statement}")
        unresolved_items.extend(f"主张 {card.id}：{item}" for item in card.remaining_unknowns)
    for card in preparation_cards:
        handling = "（本人报告已处理，但原判断未改变）" if card.handling_state == "user_reported_done" else ""
        unresolved_items.append(f"准备 {card.id}：{card.title}{handling}")
        unresolved_items.extend(f"准备 {card.id}：{item}" for item in card.remaining_unknowns)
    if not claim:
        unresolved_items.append("主张核查未形成当前结果")
    if not preparation:
        unresolved_items.append("准备检查未形成当前结果")

    next_actions = _unique(
        [card.suggested_action for card in claim_cards if card.status != "supported_limited"]
        + [card.suggested_action for card in preparation_cards]
    )
    preparation_input = preparation_run.get("input_snapshot") if preparation_run else None
    requests = (
        _request_records("video", video_run)
        + _request_records("scene", scene_run)
        + _request_records("claim", claim_run)
    )
    conclusion = {
        "complete": "视频、主张核查和准备检查均已形成当前版本结果；这只表示工作流完整，不表示整段视频正确、个人适练或场地安全。",
        "partial": "报告仅汇总当前已完成阶段；未完成或未覆盖部分必须先补充，不能据此作整体判断。",
        "failed": "视频阶段失败，报告只能记录失败状态，不能形成动作相关核查结论。",
    }[workflow_status]
    result = FormalReport(
        report_id=_report_id(input_versions, revision),
        revision=revision,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        title="适练有据 · 限定范围核查报告",
        workflow_status=workflow_status,
        boundary_statement=(
            "本报告是软件对已识别视频表述、本人已填写条件和当前场地候选的限定范围整理；"
            "不是医疗诊断、个人适练许可、动作教学、整段视频真伪判定或场地安全认证。"
        ),
        conclusion=conclusion,
        knowledge_version=knowledge.package.version,
        input_versions=input_versions,
        stages=stages,
        actions=[
            ReportAction(
                id=item.id,
                type=item.type,
                description=item.description,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                recognition=item.recognition,
                scope=item.scope,
            )
            for item in video.actions
        ],
        claim_cards=claim_cards,
        context_items=context_items,
        preparation_cards=preparation_cards,
        preparation_input=preparation_input,
        status_counts=_status_counts(claim_cards, preparation_cards),
        coverage_notes=_unique(coverage_notes),
        unresolved_items=_unique(unresolved_items),
        next_actions=next_actions,
        evidence=evidence,
        requests=requests,
    )
    markdown = render_report_markdown(result)
    return ReportRun(result, markdown, time.perf_counter() - started)


def _md(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\\\",
        "*": "\\*",
        "_": "\\_",
        "[": "\\[",
        "]": "\\]",
        "<": "&lt;",
        ">": "&gt;",
        "|": "\\|",
        "\r": " ",
        "\n": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _time_range(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "时间未知"
    return f"{int(start) // 60:02d}:{int(start) % 60:02d}—{int(end) // 60:02d}:{int(end) % 60:02d} 附近"


def render_report_markdown(report: FormalReport) -> str:
    stage_labels = {"video": "视频提取", "claim": "具体说法核查", "preparation": "准备条件检查"}
    lines = [
        f"# {_md(report.title)}",
        "",
        f"- 报告编号：`{_md(report.report_id)}`",
        f"- 生成时间：{_md(report.generated_at)}",
        f"- 工作流状态：`{_md(report.workflow_status)}`（仅表示阶段完成度）",
        f"- 资料版本：`{_md(report.knowledge_version)}`",
        "",
        "> **重要边界：** " + _md(report.boundary_statement),
        "",
        "## 本次结论",
        "",
        _md(report.conclusion),
        "",
        "## 阶段状态与版本",
        "",
        "| 阶段 | 状态 | 分析ID | 修订 | 说明 |",
        "|---|---|---|---:|---|",
    ]
    for stage in report.stages:
        lines.append(
            f"| {stage_labels[stage.name]} | `{stage.status}` | {_md(stage.analysis_id or '—')} | "
            f"{stage.revision or '—'} | {_md(stage.note)} |"
        )
    lines.extend(["", "## 本次本人条件与器材输入", ""])
    if report.preparation_input:
        conditions = report.preparation_input.get("user_conditions", {})
        lines.extend(
            [
                f"- 区域：{_md(report.preparation_input.get('area_name') or '未填写')}",
                f"- 当前不适：`{_md(conditions.get('current_discomfort', 'unknown'))}`",
                f"- 站立/坐站支撑需求：`{_md(conditions.get('support_needed', 'unknown'))}`",
                f"- 已知限制状态：`{_md(conditions.get('restriction_status', 'unknown'))}`",
                f"- 已知限制标签：{_md('、'.join(conditions.get('restrictions', [])) or '无/未填写')}",
                f"- 坐站协助需求：`{_md(conditions.get('sit_to_stand_assistance', 'not_applicable'))}`",
                f"- 本次协助安排：`{_md(conditions.get('helper_available', 'not_applicable'))}`",
            ]
        )
        if conditions.get("restriction_note"):
            lines.append("- 其他原有要求（未自动解释）：" + _md(conditions["restriction_note"]))
        equipment = report.preparation_input.get("selected_equipment", [])
        if equipment:
            for item in equipment:
                lines.append(
                    f"- 实际器材 {_md(item.get('id', '未编号'))}：候选 {_md(item.get('candidate_id', '未知'))}；"
                    f"用途 {_md('、'.join(item.get('roles', [])) or '未指定')}；"
                    f"本人属性 {_md(json.dumps(item.get('user_report', {}), ensure_ascii=False, sort_keys=True))}"
                )
        else:
            lines.append("- 实际器材：尚未指定。")
    else:
        lines.append("- 尚无准备检查输入快照。")
    lines.extend(["", "## 已识别动作", ""])
    if report.actions:
        for action in report.actions:
            lines.append(
                f"- **{_md(action.id)} · {_md(action.type)}**（{_time_range(action.start_seconds, action.end_seconds)}；"
                f"识别 `{_md(action.recognition)}`；范围 `{_md(action.scope)}`）：{_md(action.description)}"
            )
    else:
        lines.append("- 没有可列出的动作；这不表示视频中确定没有动作。")

    lines.extend(["", "## 具体说法核查", ""])
    if report.claim_cards:
        for card in report.claim_cards:
            lines.extend(
                [
                    f"### {_md(card.title)}（`{_md(card.status)}`）",
                    "",
                    f"- 核查表述：{_md(card.checked_statement)}",
                    f"- 事实引用：{_md('、'.join(card.fact_refs))}",
                    f"- 动作引用：{_md('、'.join(card.action_ids) or '未确认')}",
                    f"- 资料引用：{_md('、'.join(card.evidence_ids) or '无')}",
                    f"- 说明：{_md(card.explanation)}",
                    f"- 下一步：{_md(card.suggested_action)}",
                ]
            )
            if card.remaining_unknowns:
                lines.append("- 仍不确定：" + _md("；".join(card.remaining_unknowns)))
            lines.append("")
    else:
        lines.append("- 当前报告没有主张核查卡；请查看阶段状态与覆盖说明。")
    if report.context_items:
        lines.extend(["", "### 保留上下文或非核查项", ""])
        for item in report.context_items:
            lines.append(
                f"- {_md(item.checked_statement)}（来源 {_md('、'.join(item.fact_refs))}；"
                f"处理 `{_md(item.disposition)}`；{_md(item.explanation)}）"
            )

    lines.extend(["", "## 本人、场地和器材准备", ""])
    if report.preparation_cards:
        for card in report.preparation_cards:
            lines.extend(
                [
                    f"### {_md(card.title)}（`{_md(card.status)}`）",
                    "",
                    f"- 规则：{_md('、'.join(card.rule_ids) or '范围门槛')}",
                    f"- 事实引用：{_md('、'.join(card.fact_refs) or '无')}",
                    f"- 本人填写引用：{_md('、'.join(card.condition_refs) or '无')}",
                    f"- 资料引用：{_md('、'.join(card.evidence_ids) or '无')}",
                    f"- 说明：{_md(card.explanation)}",
                    f"- 最小行动：{_md(card.suggested_action)}",
                    f"- 操作状态：`{_md(card.handling_state)}`（不改变原判断）",
                ]
            )
            if card.remaining_unknowns:
                lines.append("- 仍不确定：" + _md("；".join(card.remaining_unknowns)))
            lines.append("")
    else:
        lines.append("- 当前报告没有准备检查卡；可能尚未检查，或当前范围未触发提示，均不代表安全许可。")

    lines.extend(["", "## 覆盖限制与未解决项", ""])
    for item in report.coverage_notes:
        lines.append("- 覆盖说明：" + _md(item))
    for item in report.unresolved_items:
        lines.append("- 未解决：" + _md(item))
    if not report.unresolved_items:
        lines.append("- 当前结构化结果未列出未解决项；这不等于没有未识别风险。")

    lines.extend(["", "## 下一步清单", ""])
    if report.next_actions:
        for item in report.next_actions:
            lines.append("- " + _md(item))
    else:
        lines.append("- 当前没有自动生成的新行动；不要据此推断可以跟练。")

    lines.extend(["", "## 本地资料来源", ""])
    if report.evidence:
        for evidence in report.evidence:
            lines.extend(
                [
                    f"### {_md(evidence.id)} · {_md(evidence.source_title)}",
                    "",
                    f"- 发布者：{_md(evidence.publisher)}",
                    f"- 本地短摘录：{_md(evidence.excerpt)}",
                    f"- 适用范围：{_md(evidence.applicability)}",
                    f"- 不支持扩展为：{_md('；'.join(evidence.exclusions))}",
                    f"- 项目核对日期：{_md(evidence.checked_at)}",
                    f"- 原始来源：[{_md(evidence.url)}]({evidence.url})",
                    "",
                ]
            )
    else:
        lines.append("- 当前报告没有证据引用。")

    lines.extend(["", "## 请求统计", ""])
    lines.append(f"报告生成阶段新增模型请求：{report.model_requests_during_report}。")
    for request in report.requests:
        lines.append(
            f"- `{request.stage}` · {_md(request.label)}：{request.elapsed_seconds:.2f} 秒；"
            f"型号 {_md(request.response_model or '未提供')}；请求ID {_md(request.request_id or '未提供')}；"
            f"usage {_md(json.dumps(request.usage, ensure_ascii=False) if request.usage else '未提供')}"
        )
    lines.extend(
        [
            "",
            "---",
            "",
            "本报告仅汇总生成时的本地会话结果。输入、场地照片、本人条件或资料版本改变后，应重新检查并重新生成报告。",
            "",
        ]
    )
    return "\n".join(lines)


def report_to_serializable(run: ReportRun) -> dict[str, Any]:
    return {
        "result": run.result.model_dump(mode="json"),
        "markdown": run.markdown,
        "elapsed_seconds": run.elapsed_seconds,
    }
