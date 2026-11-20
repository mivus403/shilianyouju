"""离线检查资料包结构与引用；不验证医学结论、不调用模型。"""

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    evidence = load("knowledge/evidence.json")
    rules = load("knowledge/rules.json")
    require(evidence["version"] == rules["version"], "资料与规则版本不一致")
    cards = evidence["cards"]
    ids = {card["id"] for card in cards}
    require(len(ids) == len(cards), "资料ID重复")
    actions = set(rules["supported_actions"])
    text_fields = (
        "id", "topic", "source_title", "publisher", "source_type", "locator",
        "excerpt", "context", "applicability", "allowed_statement", "version",
    )
    for card in cards:
        for field in text_fields:
            require(isinstance(card[field], str) and card[field].strip(), f"{card['id']}: {field}为空")
        for field in ("url", "verified_copy_url"):
            if card[field] is not None:
                url = urlparse(card[field])
                require(url.scheme == "https" and bool(url.netloc), f"{card['id']}: URL无效")
        for field in ("checked_at", "published_at", "source_last_reviewed_at", "content_date"):
            if card.get(field) is not None:
                require(date.fromisoformat(card[field]) <= date.fromisoformat(evidence["checked_at"]),
                        f"{card['id']}: 日期晚于核对日期")
        require(bool(card["exclusions"]), f"{card['id']}: 缺少排除范围")
        scope = card["match_scope"]
        require(scope["kind"] in {"general_principle", "action_specific", "environment_principle"},
                f"{card['id']}: 未知适用范围类型")
        require(bool(scope["topics"]), f"{card['id']}: 缺少主题")
        require(set(scope["actions"]) <= actions, f"{card['id']}: 动作越界")
        require(bool(scope["actions"]) == (scope["kind"] == "action_specific"),
                f"{card['id']}: 一般原则与动作专属范围混淆")

    condition_rules = rules["condition_rules"]
    require(len({rule["id"] for rule in condition_rules}) == len(condition_rules), "规则ID重复")
    prep_statuses = {"condition_gap", "needs_confirmation", None}
    for rule in condition_rules:
        require(bool(rule["actions"]) and set(rule["actions"]) <= actions, f"{rule['id']}: 动作范围无效")
        require(set(rule["evidence_ids"]) <= ids, f"{rule['id']}: 资料ID不存在")
        require(isinstance(rule["can_mark_done"], bool), f"{rule['id']}: 操作状态类型错误")
        for branch in rule["branches"]:
            require(branch["status"] in prep_statuses, f"{rule['id']}: 状态无效")
            require(bool(branch["when"]), f"{rule['id']}: 条件为空")
            require(set(branch.get("evidence_ids", [])) <= ids, f"{rule['id']}: 分支引用不存在")
            if branch["status"] is not None:
                require(bool(branch["title"]) and bool(branch["suggested_action"]), f"{rule['id']}: 缺少文案")

    checked_claims = 0
    for path in sorted((ROOT / "samples").glob("case_*/expected.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        require(fixture["is_expected_not_model_output"] is True, f"{path.name}: 缺少预期标记")
        require(fixture["knowledge_version"] == evidence["version"], f"{fixture['fixture_id']}: 样例资料版本过期")
        for claim in fixture["claim_expectations"]:
            if "expected_status" not in claim:
                continue
            require(claim["expected_status"] in rules["claim_policy"]["statuses"], "样例状态无效")
            refs = claim["expected_evidence_ids"]
            require(set(refs) <= ids, f"{claim['id']}: 引用不存在")
            if claim["expected_status"] in rules["claim_policy"]["required_evidence_statuses"]:
                require(bool(refs), f"{claim['id']}: 确定性结论缺少资料")
            require(0 <= claim["start_seconds"] <= claim["end_seconds"], f"{claim['id']}: 时间无效")
            checked_claims += 1

    print(f"PASS: {len(cards)} evidence cards, {len(condition_rules)} rule groups, {checked_claims} claim expectations.")
    print("Offline structure/reference validation only; no model calls or medical-accuracy measurement.")


if __name__ == "__main__":
    main()
