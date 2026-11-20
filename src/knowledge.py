from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


class KnowledgeError(RuntimeError):
    pass


class MatchScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    topics: list[str] = Field(min_length=1)
    actions: list[str]


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: HttpUrl
    verified_copy_url: HttpUrl | None
    source_type: str = Field(min_length=1)
    published_at: str | None
    content_date: str | None = None
    source_last_reviewed_at: str | None
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    context: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    exclusions: list[str] = Field(min_length=1)
    allowed_statement: str = Field(min_length=1)
    match_scope: MatchScope
    checked_at: str = Field(min_length=1)
    version: str = Field(min_length=1)


class EvidencePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    version: str = Field(min_length=1)
    checked_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    cards: list[EvidenceCard] = Field(min_length=1)


@dataclass(frozen=True)
class KnowledgeBase:
    package: EvidencePackage
    cards: dict[str, EvidenceCard]
    claim_policy: dict
    preparation_gates: list[str]
    condition_rules: dict[str, dict]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"无法读取资料文件 {path.name}：{exc}") from exc


def load_knowledge(project_root: Path) -> KnowledgeBase:
    try:
        package = EvidencePackage.model_validate(_read_json(project_root / "knowledge" / "evidence.json"))
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", []))
        raise KnowledgeError(f"资料结构无效（{location}）：{first.get('msg', '字段无效')}") from exc

    cards = {card.id: card for card in package.cards}
    if len(cards) != len(package.cards):
        raise KnowledgeError("资料卡 ID 重复")

    rules = _read_json(project_root / "knowledge" / "rules.json")
    if rules.get("version") != package.version:
        raise KnowledgeError("资料卡与规则版本不一致")
    policy = rules.get("claim_policy")
    if not isinstance(policy, dict):
        raise KnowledgeError("rules.json 缺少 claim_policy")
    preparation_gates = rules.get("preparation_gates")
    if not isinstance(preparation_gates, list) or not all(isinstance(item, str) for item in preparation_gates):
        raise KnowledgeError("rules.json 缺少 preparation_gates")
    condition_rule_list = rules.get("condition_rules")
    if not isinstance(condition_rule_list, list):
        raise KnowledgeError("rules.json 缺少 condition_rules")
    condition_rules = {
        item.get("id"): item
        for item in condition_rule_list
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(condition_rules) != {"R01", "R02", "R03", "R04", "R05", "R06"}:
        raise KnowledgeError("rules.json 必须且只能包含 R01—R06 六组条件规则")
    if len(condition_rules) != len(condition_rule_list):
        raise KnowledgeError("条件规则 ID 重复或结构无效")
    return KnowledgeBase(
        package=package,
        cards=cards,
        claim_policy=policy,
        preparation_gates=preparation_gates,
        condition_rules=condition_rules,
    )
