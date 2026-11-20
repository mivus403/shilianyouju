from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import streamlit as st

from src.analysis import AnalysisError, analyze_scene, analyze_video, run_to_serializable
from src.claim_check import ClaimCheckError, analyze_claims, build_claim_candidates, claim_check_to_serializable
from src import ui
from src.config import load_settings
from src.demo import DemoError, build_presets, list_cases, load_case, scene_by_id
from src.knowledge import KnowledgeError, load_knowledge
from src.media import MediaError, VideoMetadata, prepare_video, validate_image, validate_video
from src.preparation import (
    EquipmentAvailability,
    EquipmentUserReport,
    PreparationInput,
    RestrictionMatch,
    SelectedEquipment,
    UserConditions,
    evaluate_preparation,
    observation_id,
    preparation_to_serializable,
    set_handling_state,
)
from src.reporting import build_report, report_to_serializable
from src.workspace import create_session_dir, file_digest, prune_stale_sessions, remove_owned, save_owned_file


PROJECT_ROOT = Path(__file__).resolve().parent


def init_state() -> None:
    defaults = {
        "session_dir": None,
        "video_hash": None,
        "video_path": None,
        "video_bytes": None,
        "video_metadata": None,
        "video_validation_error": None,
        "video_run": None,
        "video_run_error": None,
        "video_revision": 0,
        "playback_start": 0,
        "claim_reviews": {},
        "claim_review_revision": 0,
        "claim_check_revision": 0,
        "claim_check_run": None,
        "claim_check_error": None,
        "claim_check_failure": None,
        "image_hash": None,
        "image_paths": [],
        "image_metadata": [],
        "image_validation_seconds": None,
        "image_validation_error": None,
        "scene_run": None,
        "scene_run_error": None,
        "scene_revision": 0,
        "preparation_input": None,
        "preparation_revision": 0,
        "preparation_run": None,
        "preparation_error": None,
        "preparation_form_epoch": 0,
        "report_revision": 0,
        "report_run": None,
        "report_error": None,
        "demo_case": None,
        "demo_scene_id": None,
        "demo_images": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.session_dir is None:
        prune_stale_sessions()
        st.session_state.session_dir = create_session_dir()


def clear_report_state() -> None:
    st.session_state.report_run = None
    st.session_state.report_error = None


def clear_claim_state() -> None:
    st.session_state.claim_reviews = {}
    st.session_state.claim_review_revision = 0
    st.session_state.claim_check_revision = 0
    st.session_state.claim_check_run = None
    st.session_state.claim_check_error = None
    st.session_state.claim_check_failure = None
    clear_report_state()


def clear_preparation_state(*, clear_area: bool = False, clear_matches: bool = False) -> None:
    st.session_state.preparation_run = None
    st.session_state.preparation_error = None
    current = st.session_state.preparation_input
    if current:
        updated = copy.deepcopy(current)
        if clear_area:
            updated["area_name"] = None
            updated["equipment_availability"] = {"seat": "unknown", "standing_support": "unknown"}
            updated["selected_equipment"] = []
            updated["observation_confirmations"] = {}
        if clear_matches:
            updated["restriction_matches"] = []
        st.session_state.preparation_input = updated
    st.session_state.preparation_form_epoch += 1
    clear_report_state()


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "时间未知"
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d} 附近"


def sync_video(uploaded, settings) -> None:
    if uploaded is None:
        if st.session_state.video_hash is not None:
            if st.session_state.video_path:
                old = Path(st.session_state.video_path)
                if old.exists():
                    remove_owned(old, st.session_state.session_dir)
            processing = st.session_state.session_dir / "video-processing"
            if processing.exists():
                remove_owned(processing, st.session_state.session_dir)
            st.session_state.video_hash = None
            st.session_state.video_path = None
            st.session_state.video_bytes = None
            st.session_state.video_metadata = None
            st.session_state.video_validation_error = None
            st.session_state.video_run = None
            st.session_state.video_run_error = None
            st.session_state.playback_start = 0
            clear_claim_state()
            clear_preparation_state(clear_matches=True)
        return
    data = uploaded.getvalue()
    digest = file_digest(data)
    if digest == st.session_state.video_hash:
        return
    if st.session_state.video_path:
        old = Path(st.session_state.video_path)
        if old.exists():
            remove_owned(old, st.session_state.session_dir)
    processing = st.session_state.session_dir / "video-processing"
    if processing.exists():
        remove_owned(processing, st.session_state.session_dir)
    st.session_state.video_hash = digest
    st.session_state.video_bytes = data
    st.session_state.video_run = None
    st.session_state.video_run_error = None
    st.session_state.playback_start = 0
    clear_claim_state()
    clear_preparation_state(clear_matches=True)
    st.session_state.video_revision += 1
    path = save_owned_file(st.session_state.session_dir, data, f"video-{digest[:12]}.mp4")
    st.session_state.video_path = str(path)
    try:
        st.session_state.video_metadata = validate_video(path, uploaded.name, settings.ffmpeg_path)
        st.session_state.video_validation_error = None
    except MediaError as exc:
        st.session_state.video_metadata = None
        st.session_state.video_validation_error = str(exc)


def sync_images(uploaded_files) -> None:
    if not uploaded_files:
        if st.session_state.image_hash is not None:
            for old_path in st.session_state.image_paths:
                old = Path(old_path)
                if old.exists():
                    remove_owned(old, st.session_state.session_dir)
            st.session_state.image_hash = None
            st.session_state.image_paths = []
            st.session_state.image_metadata = []
            st.session_state.image_validation_seconds = None
            st.session_state.image_validation_error = None
            st.session_state.scene_run = None
            st.session_state.scene_run_error = None
            clear_preparation_state(clear_area=True)
        return
    payloads = [item.getvalue() for item in uploaded_files]
    digest = file_digest(b"".join(bytes.fromhex(file_digest(data)) for data in payloads))
    if digest == st.session_state.image_hash:
        return
    for old_path in st.session_state.image_paths:
        old = Path(old_path)
        if old.exists():
            remove_owned(old, st.session_state.session_dir)
    st.session_state.image_hash = digest
    st.session_state.image_paths = []
    st.session_state.image_metadata = []
    st.session_state.image_validation_seconds = None
    st.session_state.scene_run = None
    st.session_state.scene_run_error = None
    clear_preparation_state(clear_area=True)
    st.session_state.scene_revision += 1
    errors: list[str] = []
    validation_started = time.perf_counter()
    for index, (uploaded, data) in enumerate(zip(uploaded_files, payloads), start=1):
        path = save_owned_file(st.session_state.session_dir, data, f"photo-{digest[:8]}-{index}.bin")
        st.session_state.image_paths.append(str(path))
        try:
            st.session_state.image_metadata.append(validate_image(path))
        except MediaError as exc:
            errors.append(f"第 {index} 张（{uploaded.name}）：{exc}")
    st.session_state.image_validation_seconds = time.perf_counter() - validation_started
    st.session_state.image_validation_error = "；".join(errors) if errors else None


def request_stats(run_data: dict) -> None:
    requests = run_data.get("requests", [])
    st.caption(f"模型请求 {len(requests)} 次；接口总耗时 {sum(item['elapsed_seconds'] for item in requests):.2f} 秒。未根据 Token 估算费用。")
    for item in requests:
        with st.expander(f"请求统计 · {item['label']}"):
            st.write(
                {
                    "耗时（秒）": round(item["elapsed_seconds"], 3),
                    "返回型号": item.get("response_model"),
                    "请求ID": item.get("request_id"),
                    "接口用量（原样）": item.get("usage") or "接口未提供",
                }
            )


def replay_button(item_id: str, start_seconds: float | None) -> None:
    if start_seconds is None:
        return
    if st.button("回看此处", key=f"replay-{item_id}"):
        st.session_state.playback_start = max(0, int(start_seconds) - 2)
        st.rerun()


def show_video_result(run_data: dict) -> None:
    result = run_data["result"]
    if result["status"] == "complete":
        st.success("视频内容已提取完成。这只是“视频里说了什么、做了什么”，是否有依据要看下一步核查。")
    elif result["status"] == "failed":
        st.error("视频请求或结构处理全部失败，没有可展示的提取事实。")
    else:
        st.warning("视频仅部分完成，请查看下方不确定信息；缺失内容不能当作已核查。")
        if result["unprocessed_ranges"]:
            st.caption(f"未处理时间范围：{result['unprocessed_ranges']}")
    media = run_data.get("media")
    if media:
        st.caption(
            f"媒体方案：{media['strategy']}；{media['part_count']} 个请求片段；"
            f"{media['source_size_bytes']:,} → {media['processed_size_bytes']:,} 字节；"
            f"处理耗时 {media['processing_seconds']:.2f} 秒。"
        )

    st.subheader("动作与器材")
    if not result["actions"]:
        st.info("模型没有返回动作；这不等于视频中确定没有动作。")
    for item in result["actions"]:
        with st.container(border=True):
            st.markdown(f"**{item['description']}**")
            st.caption(f"{format_time(item['start_seconds'])} · {ui.label(item['recognition'])} · {ui.label(item['scope'])}")
            st.write("器材：" + ("、".join(item["equipment"]) if item["equipment"] else "未提取到"))
            replay_button(item["id"], item["start_seconds"])

    st.subheader("视频说法")
    st.caption("点“回看此处”，上方播放器会跳到对应时间。")
    if not result["claims"]:
        st.info("模型没有返回说法；这不等于视频中确定没有说法。")
    for item in result["claims"]:
        with st.container(border=True):
            st.markdown(f"**{item['text']}**")
            st.caption(
                f"{format_time(item['start_seconds'])} · {ui.label(item['channel'])} · {ui.label(item['text_kind'])} · "
                f"{ui.label(item['stance'])} · {ui.label(item['recognition'])}"
            )
            replay_button(item["id"], item["start_seconds"])

    st.subheader("注意事项")
    if not result["cautions"]:
        st.info("模型没有返回注意事项；这不等于视频没有风险或已通过核查。")
    for item in result["cautions"]:
        with st.container(border=True):
            st.markdown(f"**{item['text']}**")
            st.caption(f"{format_time(item['start_seconds'])} · {ui.label(item['channel'])} · {ui.label(item['text_kind'])} · {ui.label(item['recognition'])}")
            replay_button(item["id"], item["start_seconds"])

    if result["uncertainties"]:
        st.warning("不确定信息：\n\n- " + "\n- ".join(result["uncertainties"]))
    request_stats(run_data)
    with st.expander("查看结构化视频结果"):
        st.json(result)
    with st.expander("查看模型原始回复"):
        for raw in run_data["raw_responses"]:
            repaired = "未知（解析失败）" if raw["json_repaired"] is None else ("是" if raw["json_repaired"] else "否")
            st.caption(f"{raw['label']} · 有限 JSON 修复：{repaired}")
            if raw.get("parse_error"):
                st.error(raw["parse_error"])
            st.code(raw["text"], language="json")


def show_claim_review_editor(video_result: dict) -> None:
    st.subheader("回看与更正识别")
    st.caption("更正只保存在本机当前会话，不覆盖模型原始记录；保存后旧核查结果会失效，重新核查才会调用接口。")
    facts = [("说法", item) for item in video_result["claims"]]
    facts.extend(("注意事项", item) for item in video_result["cautions"])
    if not facts:
        st.info("当前没有已提取的说法或注意事项。若视频有音轨但语音为空，不能据此认定视频没有说法。")
        return

    reviews = st.session_state.claim_reviews
    inferred = {
        candidate.fact_refs[0]: candidate.action_ids
        for candidate in build_claim_candidates(video_result, reviews)
    }
    action_options = [item["id"] for item in video_result["actions"]]
    action_labels = {
        item["id"]: f"{item['id']} · {item['type']} · {format_time(item['start_seconds'])}"
        for item in video_result["actions"]
    }
    stance_labels = {
        None: "未确认/保持注意事项原状态",
        "endorsed": "视频直接主张",
        "rejected": "视频明确反对",
        "quoted_only": "只转述他人说法",
        "unclear": "立场不清",
    }

    for source_label, item in facts:
        review = reviews.get(item["id"])
        marker = " · 已有本地更正" if review else ""
        with st.expander(f"{source_label} {item['id']} · {format_time(item['start_seconds'])}{marker}"):
            st.write("原始识别：" + item["text"])
            st.caption(
                f"来源：{item['channel']} · 识别：{item['recognition']}"
                + (f" · 原始立场：{item['stance']}" if "stance" in item else "")
            )
            replay_button(f"review-{item['id']}", item["start_seconds"])
            baseline_text = item.get("corrected_text") or item["text"]
            current_text = review.get("corrected_text") if review and review.get("corrected_text") else baseline_text
            current_stance = review.get("stance") if review else item.get("stance")
            current_actions = review.get("action_ids") if review and "action_ids" in review else inferred[item["id"]]
            with st.form(f"review-form-{st.session_state.video_revision}-{item['id']}"):
                corrected_text = st.text_area("核查用完整文字", value=current_text)
                recognition_confirmed = st.checkbox(
                    "我已回看，确认上方完整文字及否定范围识别准确",
                    value=bool((review or {}).get("recognition_confirmed", False)),
                    help="只确认识别文字，不代表这项健康说法正确。未确认时保留原识别状态。",
                )
                stance_options = [None, "endorsed", "rejected", "quoted_only", "unclear"]
                stance = st.selectbox(
                    "确认说话者立场",
                    options=stance_options,
                    index=stance_options.index(current_stance) if current_stance in stance_options else 0,
                    format_func=lambda value: stance_labels[value],
                )
                linked_actions = st.multiselect(
                    "关联动作（不确定可留空）",
                    options=action_options,
                    default=current_actions,
                    format_func=lambda action_id: action_labels[action_id],
                )
                save = st.form_submit_button("保存本地复核")
            if save:
                if not corrected_text.strip():
                    st.error("核查用文字不能为空。")
                else:
                    updated = dict(reviews)
                    updated[item["id"]] = {
                        "corrected_text": corrected_text.strip() if corrected_text.strip() != baseline_text else None,
                        "stance": stance,
                        "recognition_confirmed": recognition_confirmed,
                        "action_ids": linked_actions,
                        "version": int((review or {}).get("version", 0)) + 1,
                    }
                    st.session_state.claim_reviews = updated
                    st.session_state.claim_review_revision += 1
                    st.session_state.claim_check_run = None
                    st.session_state.claim_check_error = None
                    st.session_state.claim_check_failure = None
                    clear_report_state()
                    st.rerun()
            if review and st.button("撤销这条本地复核", key=f"undo-review-{item['id']}"):
                updated = dict(reviews)
                updated.pop(item["id"], None)
                st.session_state.claim_reviews = updated
                st.session_state.claim_review_revision += 1
                st.session_state.claim_check_run = None
                st.session_state.claim_check_error = None
                st.session_state.claim_check_failure = None
                clear_report_state()
                st.rerun()


def show_claim_check_result(run_data: dict, video_result: dict, knowledge) -> None:
    result = run_data["result"]
    st.success("具体说法与本地资料对照完成。这不是整段视频正确、个人适练或医疗建议。")
    if result["coverage_notes"]:
        st.warning("识别与核查覆盖仍有限：\n\n- " + "\n- ".join(result["coverage_notes"]))
    if result["validation_notes"]:
        st.warning("引用校验处理：\n\n- " + "\n- ".join(result["validation_notes"]))

    fact_lookup = {
        item["id"]: item
        for group in (video_result["claims"], video_result["cautions"])
        for item in group
    }

    def render_card(card: dict) -> None:
        ui.status_marker(card["status"])
        label, color = ui.CLAIM_STATUS[card["status"]]
        st.markdown(ui.chip(label, color) + f"**{card['title']}**", unsafe_allow_html=True)
        st.write(card["checked_statement"])
        st.caption(
            f"事实引用：{'、'.join(card['fact_refs'])}"
            + (f" · 动作引用：{'、'.join(card['action_ids'])}" if card["action_ids"] else " · 动作关联未确认")
        )
        st.write(card["explanation"])
        label = "资料允许的有限表述" if card["evidence_ids"] else "下一步"
        st.info(f"{label}：{card['suggested_action']}")
        if card["remaining_unknowns"]:
            st.warning("仍不确定：" + "；".join(card["remaining_unknowns"]))
        first_fact = fact_lookup.get(card["fact_refs"][0])
        if first_fact:
            replay_button(f"claim-card-{card['id']}", first_fact["start_seconds"])
        for evidence_id in card["evidence_ids"]:
            evidence = knowledge.cards[evidence_id]
            with st.expander(f"查看依据 {evidence_id} · {evidence.source_title}"):
                st.write(f"发布者：{evidence.publisher}")
                st.write(f"短摘录：{evidence.excerpt}")
                st.write(f"适用范围：{evidence.applicability}")
                st.write("不支持扩展为：" + "；".join(evidence.exclusions))
                st.caption(f"项目核对日期：{evidence.checked_at} · 资料包版本：{result['knowledge_version']}")
                st.markdown(f"[打开原始来源]({evidence.url})")
                if evidence.verified_copy_url:
                    st.markdown(f"[打开已核对备用转载]({evidence.verified_copy_url})")

    priority = {"recognition_needed": 0, "conflicts": 1, "overstated": 2, "insufficient": 3, "supported_limited": 4}
    ordered = sorted(result["cards"], key=lambda card: priority[card["status"]])
    main_cards = [card for card in ordered if card["status"] != "supported_limited"]
    supported_cards = [card for card in ordered if card["status"] == "supported_limited"]
    st.subheader("具体说法核查卡")
    ui.stat_row(
        [
            (ui.CLAIM_STATUS[key][0], sum(1 for card in ordered if card["status"] == key), ui.CLAIM_STATUS[key][1])
            for key in ui.CLAIM_STATUS
            if any(card["status"] == key for card in ordered)
        ]
    )
    if not ordered:
        st.info("本次没有形成主张核查卡；这不等于视频没有问题。请查看保留上下文和覆盖说明。")
    for card in main_cards:
        with st.container(border=True):
            render_card(card)
    if supported_cards:
        with st.expander(f"有限范围有依据的表述（{len(supported_cards)}）"):
            for card in supported_cards:
                with st.container(border=True):
                    render_card(card)

    if result["context_items"]:
        with st.expander(f"保留上下文或非健康说法（{len(result['context_items'])}）"):
            for item in result["context_items"]:
                st.write(f"- {item['checked_statement']}（来源：{'、'.join(item['fact_refs'])}；{item['explanation']}）")

    request_stats(run_data)
    with st.expander("查看结构化主张核查结果"):
        st.json(result)
    with st.expander("查看主张核查模型原始回复"):
        raw = run_data["raw_responses"][0]
        st.caption("有限 JSON 修复：" + ("是" if raw["json_repaired"] else "否"))
        st.code(raw["text"], language="json")


def show_scene_result(run_data: dict) -> None:
    result = run_data["result"]
    if result["status"] == "failed":
        st.error("场地模型回复未通过结构或事实边界校验，没有可展示的提取事实。")
    else:
        st.success("场地图片内容提取完成。结果是候选观察，不是器材稳固性或场地安全认证。")
    st.subheader("可见物品")
    for item in result["visible_items"]:
        st.write(f"- {item['photo_id']} · {item['category']}：{item['description']}（{item['recognition']}）")
    if not result["visible_items"]:
        st.info("没有返回可见物品；不表示照片中确定没有物品。")

    st.subheader("器材候选与属性")
    for item in result["equipment_candidates"]:
        with st.container(border=True):
            st.markdown(f"**{item['candidate_id']} · {item['type']}**")
            st.write(item["description"])
            st.caption("来源照片：" + "、".join(item["photo_ids"]))
            for attribute in item["attributes"]:
                st.write(f"- {attribute['name']}：{attribute['value']}；{attribute['evidence']}（{attribute['recognition']}）")
    st.subheader("场地观察")
    for item in result["observations"]:
        target = f" · {item['equipment_candidate_id']}" if item["equipment_candidate_id"] else ""
        st.write(
            f"- {item['photo_id']}{target} · {item['kind']} · {item['location']}："
            f"{item['description']}（{item['recognition']}）"
        )
    if not result["observations"]:
        st.info("模型没有返回场地观察；不表示场地没有需要确认的事项。")
    if result["uncertainties"]:
        st.warning("无法确认：\n\n- " + "\n- ".join(result["uncertainties"]))
    request_stats(run_data)
    with st.expander("查看结构化场地结果"):
        st.json(result)
    with st.expander("查看模型原始回复"):
        if run_data["raw_responses"][0].get("parse_error"):
            st.error(run_data["raw_responses"][0]["parse_error"])
        st.code(run_data["raw_responses"][0]["text"], language="json")


def _choice_index(options: list, current) -> int:
    return options.index(current) if current in options else 0


def show_preparation_form(video_result: dict, scene_result: dict | None, knowledge) -> None:
    active_actions = [
        item
        for item in video_result.get("actions", [])
        if item.get("scope") == "supported" and item.get("recognition") == "clear"
    ]
    has_sit = any(item["type"] == "sit_to_stand" for item in active_actions)
    has_calf = any(item["type"] == "supported_calf_raise" for item in active_actions)
    has_standing = has_sit or has_calf
    prior = st.session_state.preparation_input or {}
    prior_conditions = prior.get("user_conditions", {})
    prior_availability = prior.get("equipment_availability", {})
    prior_equipment = prior.get("selected_equipment", [])
    prior_confirmations = prior.get("observation_confirmations", {})
    prior_matches = {
        (item["action_id"], item["restriction"]): item["user_confirmation"]
        for item in prior.get("restriction_matches", [])
    }
    answer_labels = {
        "unknown": "不确定/未填写",
        "yes": "是",
        "no": "否",
        "not_applicable": "不适用",
    }
    yes_no_unknown = ["unknown", "yes", "no"]
    restriction_labels = {
        "avoid_repeated_sit_to_stand": "避免反复坐站",
        "avoid_unsupported_standing": "避免无支撑站立",
        "avoid_calf_raise": "避免踮脚/提踵",
        "avoid_arm_raise": "避免上肢抬举",
    }
    action_labels = {
        item["id"]: f"{item['id']} · {item['type']} · {format_time(item['start_seconds'])}"
        for item in active_actions
    }
    scene_complete = bool(scene_result and scene_result.get("status") == "complete")
    candidates = scene_result.get("equipment_candidates", []) if scene_complete else []
    candidate_lookup = {item["candidate_id"]: item for item in candidates}
    candidate_ids = list(candidate_lookup)

    def prior_candidate_for(role: str):
        for equipment in prior_equipment:
            if role in equipment.get("roles", []) and equipment.get("candidate_id") in candidate_lookup:
                return equipment["candidate_id"]
        return None

    def prior_report_for(candidate_id: str) -> dict:
        for equipment in prior_equipment:
            if equipment.get("candidate_id") == candidate_id:
                return equipment.get("user_report", {})
        return {}

    epoch = st.session_state.preparation_form_epoch
    st.caption(
        "本人条件、医嘱备注、器材选择和场地确认只在本机会话中运行 R01—R06，"
        "不会发送给模型。默认均为不确定，不会预选“没有问题”。"
    )
    with st.form(f"preparation-form-{epoch}"):
        st.subheader("本人已知条件")
        current_discomfort = st.selectbox(
            "目前是否有不适？",
            yes_no_unknown,
            index=_choice_index(yes_no_unknown, prior_conditions.get("current_discomfort", "unknown")),
            format_func=lambda value: answer_labels[value],
            key=f"prep-{epoch}-discomfort",
        )
        if has_standing:
            support_needed = st.selectbox(
                "站立或坐站时需要支撑吗？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, prior_conditions.get("support_needed", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-support-needed",
            )
        else:
            support_needed = "not_applicable"

        restriction_status_options = ["unknown", "none_known", "has_restrictions"]
        restriction_status_labels = {
            "unknown": "不确定/未填写",
            "none_known": "无已知限制",
            "has_restrictions": "有已被明确告知的限制",
        }
        restriction_status = st.selectbox(
            "是否曾被明确告知活动限制？",
            restriction_status_options,
            index=_choice_index(restriction_status_options, prior_conditions.get("restriction_status", "unknown")),
            format_func=lambda value: restriction_status_labels[value],
            key=f"prep-{epoch}-restriction-status",
        )
        restrictions: list[str] = []
        restriction_note = None
        restriction_matches: list[RestrictionMatch] = []
        if restriction_status == "has_restrictions":
            restrictions = st.multiselect(
                "选择已被明确告知的限制",
                options=list(restriction_labels),
                default=[item for item in prior_conditions.get("restrictions", []) if item in restriction_labels],
                format_func=lambda value: restriction_labels[value],
                key=f"prep-{epoch}-restrictions",
            )
            restriction_note_value = st.text_input(
                "其他原有要求（只在本地展示，不由模型解释）",
                value=prior_conditions.get("restriction_note") or "",
                key=f"prep-{epoch}-restriction-note",
            )
            restriction_note = restriction_note_value.strip() or None
            relevant_types = {
                "avoid_repeated_sit_to_stand": {"sit_to_stand"},
                "avoid_unsupported_standing": {"sit_to_stand", "supported_calf_raise"},
                "avoid_calf_raise": {"supported_calf_raise"},
                "avoid_arm_raise": {"seated_arm_movement"},
                "other_note": {"seated_arm_movement", "sit_to_stand", "supported_calf_raise"},
            }
            match_labels = {"unknown": "不确定", "yes": "确认涉及", "no": "确认不涉及"}
            match_restrictions = list(restrictions) + (["other_note"] if restriction_note else [])
            for restriction in match_restrictions:
                for action in active_actions:
                    if action["type"] not in relevant_types[restriction]:
                        continue
                    if restriction == "avoid_calf_raise" and action["type"] == "supported_calf_raise":
                        continue
                    restriction_label = restriction_labels.get(restriction, "其他原有要求")
                    confirmation = st.selectbox(
                        f"{restriction_label}是否涉及 {action_labels[action['id']]}？",
                        yes_no_unknown,
                        index=_choice_index(
                            yes_no_unknown,
                            prior_matches.get((action["id"], restriction), "unknown"),
                        ),
                        format_func=lambda value: match_labels[value],
                        key=f"prep-{epoch}-match-{action['id']}-{restriction}",
                    )
                    restriction_matches.append(
                        RestrictionMatch(
                            action_id=action["id"],
                            restriction=restriction,
                            user_confirmation=confirmation,
                        )
                    )

        if has_sit:
            assistance = st.selectbox(
                "坐站转换是否需要别人帮助？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, prior_conditions.get("sit_to_stand_assistance", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-assistance",
            )
            if assistance == "yes":
                helper = st.selectbox(
                    "这次是否有人可以协助？",
                    yes_no_unknown,
                    index=_choice_index(yes_no_unknown, prior_conditions.get("helper_available", "unknown")),
                    format_func=lambda value: answer_labels[value],
                    key=f"prep-{epoch}-helper",
                )
            else:
                helper = "not_applicable"
        else:
            assistance = "not_applicable"
            helper = "not_applicable"

        st.subheader("本次区域和实际器材")
        area_name_value = st.text_input(
            "您准备在哪个区域活动？（可留空）",
            value=prior.get("area_name") or "",
            key=f"prep-{epoch}-area-name",
        )
        if not scene_complete:
            st.info("当前没有成功的场地图片提取结果。可先检查本人条件；器材和照片分支会明确显示尚未检查。")

        if has_sit:
            seat_availability = st.selectbox(
                "本次坐站座椅是否已准备？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, prior_availability.get("seat", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-seat-availability",
            )
        else:
            seat_availability = "not_applicable"
        support_relevant = has_calf or (has_sit and support_needed != "no")
        if support_relevant:
            support_availability = st.selectbox(
                "本次实际支撑是否已准备？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, prior_availability.get("standing_support", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-support-availability",
            )
        else:
            support_availability = "not_applicable"

        candidate_options = [None] + candidate_ids

        def candidate_label(value):
            if value is None:
                return "尚未指定"
            return f"{value} · {candidate_lookup[value]['type']} · {candidate_lookup[value]['description']}"

        selected_roles: dict[str, list[str]] = {}
        if has_sit and scene_complete and seat_availability != "no":
            seat_candidate = st.selectbox(
                "哪件场地候选是本次实际座椅？",
                candidate_options,
                index=_choice_index(candidate_options, prior_candidate_for("seat")),
                format_func=candidate_label,
                key=f"prep-{epoch}-seat-candidate",
            )
            if seat_candidate:
                selected_roles.setdefault(seat_candidate, []).append("seat")
        if support_relevant and scene_complete and support_availability != "no":
            support_candidate = st.selectbox(
                "哪件场地候选是本次实际支撑？",
                candidate_options,
                index=_choice_index(candidate_options, prior_candidate_for("standing_support")),
                format_func=candidate_label,
                key=f"prep-{epoch}-support-candidate",
            )
            if support_candidate:
                selected_roles.setdefault(support_candidate, []).append("standing_support")

        selected_equipment: list[SelectedEquipment] = []
        for candidate_id, roles in selected_roles.items():
            candidate = candidate_lookup[candidate_id]
            candidate_index = candidate_ids.index(candidate_id) + 1
            previous_report = prior_report_for(candidate_id)
            st.markdown(f"**确认实际器材：{candidate_id}（用途：{'、'.join(roles)}）**")
            has_wheels = st.selectbox(
                "本人确认：是否有轮？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, previous_report.get("has_wheels", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-equipment-{candidate_index}-wheels",
            )
            stability = st.selectbox(
                "本人是否已确认它稳固？（不要靠试坐、摇晃或试做验证）",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, previous_report.get("stability_confirmed", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-equipment-{candidate_index}-stable",
            )
            non_slip = st.selectbox(
                "本人是否已确认它在本次摆放表面不会滑动？",
                yes_no_unknown,
                index=_choice_index(yes_no_unknown, previous_report.get("non_slip_confirmed", "unknown")),
                format_func=lambda value: answer_labels[value],
                key=f"prep-{epoch}-equipment-{candidate_index}-non-slip",
            )
            selected_equipment.append(
                SelectedEquipment(
                    id=f"eq-{candidate_index}",
                    candidate_id=candidate_id,
                    type=candidate["type"],
                    photo_ids=candidate["photo_ids"],
                    roles=roles,
                    user_report=EquipmentUserReport(
                        has_wheels=has_wheels,
                        stability_confirmed=stability,
                        non_slip_confirmed=non_slip,
                    ),
                )
            )

        observation_confirmations: dict[str, str] = {}
        if scene_complete:
            relevant_observations = [
                (index, item)
                for index, item in enumerate(scene_result.get("observations", []), start=1)
                if item.get("kind") in {"possible_obstacle", "view_insufficient"}
            ]
            if relevant_observations:
                st.markdown("**确认场地观察与本次活动位置的关系**")
            confirmation_labels = {
                "unknown": "尚未确认",
                "confirmed": "与本次活动位置/路线有关",
                "rejected": "与本次活动位置/路线无关",
            }
            for index, item in relevant_observations:
                item_id = observation_id(scene_result, index)
                confirmation_options = ["unknown", "confirmed", "rejected"]
                confirmation = st.selectbox(
                    f"{item['photo_id']} · {item['location']}：{item['description']}",
                    confirmation_options,
                    index=_choice_index(confirmation_options, prior_confirmations.get(item_id, "unknown")),
                    format_func=lambda value: confirmation_labels[value],
                    key=f"prep-{epoch}-observation-{index}",
                )
                observation_confirmations[item_id] = confirmation

        submitted = st.form_submit_button("检查准备条件（仅本地）", type="primary", disabled=knowledge is None)

    if submitted:
        next_revision = st.session_state.preparation_revision + 1
        clear_report_state()
        try:
            preparation_input = PreparationInput(
                revision=next_revision,
                area_name=area_name_value.strip() or None,
                user_conditions=UserConditions(
                    revision=next_revision,
                    support_needed=support_needed,
                    current_discomfort=current_discomfort,
                    restriction_status=restriction_status,
                    restrictions=restrictions,
                    restriction_note=restriction_note,
                    sit_to_stand_assistance=assistance,
                    helper_available=helper,
                ),
                equipment_availability=EquipmentAvailability(
                    seat=seat_availability,
                    standing_support=support_availability,
                ),
                selected_equipment=selected_equipment,
                observation_confirmations=observation_confirmations,
                restriction_matches=restriction_matches,
            )
            run = evaluate_preparation(
                video_result,
                scene_result,
                preparation_input,
                knowledge,
                revision=next_revision,
            )
            st.session_state.preparation_revision = next_revision
            st.session_state.preparation_input = preparation_input.model_dump(mode="json")
            st.session_state.preparation_run = preparation_to_serializable(run)
            st.session_state.preparation_error = None
            clear_report_state()
        except (AnalysisError, ValueError) as exc:
            st.session_state.preparation_run = None
            st.session_state.preparation_error = str(exc)


def show_preparation_result(run_data: dict, video_result: dict, knowledge) -> None:
    result = run_data["result"]
    if result["status"] == "failed":
        st.error("准备检查未运行：" + "；".join(result["coverage_notes"]))
        return
    if result["status"] == "partial":
        st.warning("准备检查只覆盖了当前可用输入；缺失分支不会被当作没有问题。")
    else:
        st.success("本地准备规则已执行完成。complete 只表示规则运行完整，不表示个人适练或场地安全。")
    st.warning("覆盖说明：\n\n- " + "\n- ".join(result["coverage_notes"]))
    st.caption(
        f"本地规则耗时 {run_data['elapsed_seconds']:.4f} 秒；模型请求 {run_data['model_requests']} 次。"
        "本人条件与器材确认未发送到模型。"
    )

    action_lookup = {item["id"]: item for item in video_result.get("actions", [])}
    priority = {"condition_gap": 0, "needs_confirmation": 1, "unsupported": 2}
    cards = sorted(result["cards"], key=lambda card: priority[card["status"]])
    st.subheader("准备检查卡")
    if not cards:
        st.info("当前填写与已识别范围没有触发提示；这不是“可以跟练”或场地安全认证。")
    ui.stat_row(
        [
            (ui.PREPARATION_STATUS[key][0], sum(1 for card in cards if card["status"] == key), ui.PREPARATION_STATUS[key][1])
            for key in ui.PREPARATION_STATUS
            if any(card["status"] == key for card in cards)
        ]
    )
    for card in cards:
        with st.container(border=True):
            ui.status_marker(card["status"])
            label, color = ui.PREPARATION_STATUS[card["status"]]
            st.markdown(ui.chip(label, color) + f"**{card['title']}**", unsafe_allow_html=True)
            st.caption(
                "本地规则"
                + (f" · 规则：{'、'.join(card['rule_ids'])}" if card["rule_ids"] else " · 范围门槛")
                + f" · 事实引用：{'、'.join(card['fact_refs']) or '无'}"
            )
            st.write(card["explanation"])
            st.info("最小行动：" + card["suggested_action"])
            if card["remaining_unknowns"]:
                st.warning("仍不确定：" + "；".join(card["remaining_unknowns"]))
            if card["condition_refs"]:
                st.caption("本人填写引用：" + "、".join(card["condition_refs"]))
            for fact_ref in card["fact_refs"]:
                action = action_lookup.get(fact_ref)
                if action:
                    replay_button(f"preparation-{card['id']}-{fact_ref}", action["start_seconds"])
                    break
            for evidence_id in card["evidence_ids"]:
                evidence = knowledge.cards[evidence_id]
                with st.expander(f"查看依据 {evidence_id} · {evidence.source_title}"):
                    st.write(f"发布者：{evidence.publisher}")
                    st.write(f"短摘录：{evidence.excerpt}")
                    st.write(f"适用范围：{evidence.applicability}")
                    st.write("不支持扩展为：" + "；".join(evidence.exclusions))
                    st.caption(f"项目核对日期：{evidence.checked_at} · 资料包版本：{result['knowledge_version']}")
                    st.markdown(f"[打开原始来源]({evidence.url})")
            if card["can_mark_done"]:
                if card["handling_state"] == "user_reported_done":
                    st.caption("操作状态：本人报告已处理；原条件判断保持不变。")
                    new_state = "open"
                    button_label = "改回尚未处理"
                else:
                    st.caption("操作状态：尚未报告处理。该按钮不会把条件缺口改成通过。")
                    new_state = "user_reported_done"
                    button_label = "本人报告已处理"
                if st.button(button_label, key=f"handling-{card['id']}-{card['handling_state']}"):
                    try:
                        set_handling_state(st.session_state.preparation_run, card["id"], new_state)
                        clear_report_state()
                        st.rerun()
                    except AnalysisError as exc:
                        st.error(str(exc))

    with st.expander("查看结构化准备检查结果"):
        st.json(result)
    with st.expander("查看本地输入快照"):
        st.json(run_data["input_snapshot"])


def show_formal_report(run_data: dict) -> None:
    report = run_data["result"]
    workflow_labels = {
        "complete": "三个阶段均有当前版本结果",
        "partial": "报告仅覆盖部分阶段或范围",
        "failed": "视频阶段失败，只记录失败状态",
    }
    if report["workflow_status"] == "complete":
        st.success("报告已生成：" + workflow_labels[report["workflow_status"]] + "。这不是个人适练结论。")
    elif report["workflow_status"] == "partial":
        st.warning("报告已生成：" + workflow_labels[report["workflow_status"]] + "，不能据此作整体判断。")
    else:
        st.error("报告已生成：" + workflow_labels[report["workflow_status"]] + "。")
    st.error("重要边界：" + report["boundary_statement"])
    st.write(report["conclusion"])
    st.caption(
        f"报告编号：{report['report_id']} · 资料版本：{report['knowledge_version']} · "
        f"本地生成耗时 {run_data['elapsed_seconds']:.4f} 秒 · 报告生成新增模型请求 0 次"
    )

    stage_labels = {"video": "视频提取", "claim": "具体说法核查", "preparation": "准备条件检查"}
    st.subheader("阶段状态")
    st.table(
        [
            {
                "阶段": stage_labels[item["name"]],
                "状态": item["status"],
                "分析ID": item["analysis_id"] or "—",
                "修订": item["revision"] or "—",
                "说明": item["note"],
            }
            for item in report["stages"]
        ]
    )
    if report["status_counts"]:
        st.write("核查卡状态计数：", report["status_counts"])

    st.subheader("未解决项")
    if report["unresolved_items"]:
        for item in report["unresolved_items"]:
            st.write("- " + item)
    else:
        st.info("当前结构化结果未列出未解决项；这不等于没有未识别风险。")
    if report["next_actions"]:
        st.subheader("下一步清单")
        for item in report["next_actions"]:
            st.write("- " + item)

    with st.expander("覆盖限制"):
        for item in report["coverage_notes"]:
            st.write("- " + item)
    with st.expander("报告采用的本地资料来源"):
        if not report["evidence"]:
            st.write("当前报告没有证据引用。")
        for evidence in report["evidence"]:
            st.markdown(f"**{evidence['id']} · {evidence['source_title']}**")
            st.write(f"{evidence['publisher']} · 项目核对日期 {evidence['checked_at']}")
            st.write("短摘录：" + evidence["excerpt"])
            st.write("适用范围：" + evidence["applicability"])
            st.markdown(f"[打开原始来源]({evidence['url']})")
    with st.expander("预览完整 Markdown 报告"):
        st.markdown(run_data["markdown"])
    with st.expander("查看结构化报告"):
        st.json(report)

    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "下载 Markdown 报告",
        data=run_data["markdown"].encode("utf-8"),
        file_name=f"{report['report_id']}.md",
        mime="text/markdown; charset=utf-8",
        key=f"download-report-md-{report['report_id']}",
    )
    st.download_button(
        "下载结构化 JSON",
        data=json_bytes,
        file_name=f"{report['report_id']}.json",
        mime="application/json",
        key=f"download-report-json-{report['report_id']}",
    )


def reset_workspace() -> None:
    sync_video(None, settings)
    sync_images(None)
    st.session_state.video_run_error = None
    st.session_state.scene_run_error = None
    st.session_state.demo_case = None
    st.session_state.demo_scene_id = None
    st.session_state.demo_images = []
    st.session_state.playback_start = 0


def install_demo_scene(case: dict, scene_id: str | None) -> None:
    for old_path in st.session_state.image_paths:
        old = Path(old_path)
        if old.exists():
            remove_owned(old, st.session_state.session_dir)
    st.session_state.image_paths = []
    st.session_state.image_metadata = []
    st.session_state.demo_images = []
    st.session_state.scene_run_error = None
    st.session_state.image_validation_error = None
    scene = scene_by_id(case, scene_id)
    clear_preparation_state(clear_area=True)
    if scene is None:
        st.session_state.image_hash = None
        st.session_state.scene_run = None
        st.session_state.demo_scene_id = None
        return
    for index, relative in enumerate(scene["files"], start=1):
        data = (PROJECT_ROOT / "samples" / relative).read_bytes()
        path = save_owned_file(st.session_state.session_dir, data, f"photo-demo-{scene['id']}-{index}.bin")
        st.session_state.image_paths.append(str(path))
        st.session_state.image_metadata.append(validate_image(path))
        st.session_state.demo_images.append(data)
    st.session_state.image_hash = f"demo-{case['id']}-{scene['id']}"
    st.session_state.scene_run = scene["run"]
    st.session_state.scene_revision += 1
    st.session_state.demo_scene_id = scene["id"]


def load_demo_case(case_id: str) -> None:
    case = load_case(PROJECT_ROOT, case_id)
    reset_workspace()
    video_file = PROJECT_ROOT / "samples" / case["video_file"]
    data = video_file.read_bytes()
    digest = file_digest(data)
    path = save_owned_file(st.session_state.session_dir, data, f"video-{digest[:12]}.mp4")
    st.session_state.video_hash = digest
    st.session_state.video_bytes = data
    st.session_state.video_path = str(path)
    duration = case["video_run"]["result"]["duration_seconds"]
    try:
        metadata = validate_video(path, video_file.name, settings.ffmpeg_path)
    except MediaError:
        metadata = VideoMetadata(duration, len(data), "mp4", "h264", 960, 540, "present", "aac", 0)
    st.session_state.video_metadata = metadata
    st.session_state.video_revision += 1
    st.session_state.video_run = case["video_run"]
    st.session_state.claim_check_run = case["claim_run"]
    st.session_state.demo_case = case_id
    install_demo_scene(case, scene_by_id(case, None)["id"] if case["scenes"] else None)


def apply_preset(case: dict, preset) -> None:
    if preset.scene_id and preset.scene_id != st.session_state.demo_scene_id:
        install_demo_scene(case, preset.scene_id)
    next_revision = st.session_state.preparation_revision + 1
    preparation_input = preset.preparation.model_copy(deep=True)
    preparation_input.revision = next_revision
    preparation_input.user_conditions.revision = next_revision
    scene_result = st.session_state.scene_run["result"] if st.session_state.scene_run else None
    clear_report_state()
    try:
        run = evaluate_preparation(
            st.session_state.video_run["result"], scene_result, preparation_input, knowledge, revision=next_revision
        )
        st.session_state.preparation_revision = next_revision
        st.session_state.preparation_input = preparation_input.model_dump(mode="json")
        st.session_state.preparation_run = preparation_to_serializable(run)
        st.session_state.preparation_error = None
    except (AnalysisError, ValueError) as exc:
        st.session_state.preparation_run = None
        st.session_state.preparation_error = str(exc)
    st.session_state.preparation_form_epoch += 1


st.set_page_config(
    page_title="\u9002\u7EC3\u6709\u636E \u00B7 \u89C6\u9891\u8BF4\u6CD5\u6838\u67E5\u4E0E\u8DDF\u7EC3\u51C6\u5907",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_state()
settings = load_settings(PROJECT_ROOT)
knowledge = None
knowledge_error = None
try:
    knowledge = load_knowledge(PROJECT_ROOT)
except KnowledgeError as exc:
    knowledge_error = str(exc)
ui.inject_css()

demo_case = None
if st.session_state.demo_case:
    try:
        demo_case = load_case(PROJECT_ROOT, st.session_state.demo_case)
    except DemoError:
        reset_workspace()
demo_cases = list_cases(PROJECT_ROOT)

with st.sidebar:
    st.markdown("### \U0001fa7a \u9002\u7EC3\u6709\u636E")
    st.caption("\u65E0\u9700 API Key\uFF0C\u5148\u770B\u6548\u679C")
    for item in demo_cases:
        active = bool(demo_case and demo_case["id"] == item["id"])
        if st.button(("\u2705 " if active else "\u25B6 ") + item["title"], key=f"load-{item['id']}", help=item["tagline"]):
            load_demo_case(item["id"])
            st.rerun()
    if demo_case and st.button("\u21A9 \u9000\u51FA\u793A\u4F8B\uFF0C\u7528\u6211\u81EA\u5DF1\u7684\u6587\u4EF6", key="exit-demo"):
        reset_workspace()
        st.rerun()
    st.divider()
    st.markdown("**\u8FD0\u884C\u72B6\u6001**")
    st.write(("\U0001f7e2 " if settings.api_key else "\u26AA ") + ("\u5DF2\u914D\u7F6E API Key" if settings.api_key else "\u672A\u914D\u7F6E API Key\uFF08\u4EC5\u793A\u4F8B\u6A21\u5F0F\uFF09"))
    st.write(("\U0001f7e2 " if settings.ffmpeg_path else "\U0001f534 ") + ("FFmpeg \u5C31\u7EEA" if settings.ffmpeg_path else "\u672A\u627E\u5230 FFmpeg"))
    st.write(("\U0001f7e2 " if knowledge else "\U0001f534 ") + (f"\u8D44\u6599\u5E93 {len(knowledge.cards)} \u5F20\u5361" if knowledge else "\u8D44\u6599\u5E93\u4E0D\u53EF\u7528"))
    if settings.missing_model_config:
        st.caption("\u4E0A\u4F20\u81EA\u5DF1\u7684\u89C6\u9891\u9700\u8981\u914D\u7F6E\uFF1A" + "\u3001".join(settings.missing_model_config) + "\u3002\u89C1 README\u3002")
    if knowledge_error:
        st.error("\u672C\u5730\u8D44\u6599\u5E93\u4E0D\u53EF\u7528\uFF1A" + knowledge_error)
    with st.expander("\u914D\u7F6E\u8BE6\u60C5"):
        st.write({"\u6A21\u578B": settings.model, "\u670D\u52A1\u4E3B\u673A": settings.base_url_host or "\u672A\u914D\u7F6E", "FFmpeg": settings.ffmpeg_path or "\u672A\u627E\u5230"})
        if knowledge:
            st.write({"\u8D44\u6599\u5305": knowledge.package.version, "\u6838\u5BF9\u65E5\u671F": knowledge.package.checked_at})
    with st.expander("\u4F7F\u7528\u5C0F\u63D0\u793A"):
        st.markdown(
            "- \u793A\u4F8B\u7ED3\u679C\u662F\u9884\u5148\u4FDD\u5B58\u7684**\u771F\u5B9E\u6A21\u578B\u8FD0\u884C\u7ED3\u679C**\uFF0C\u9875\u9762\u4F7F\u7528\u5B83\u4EEC\u65F6\u4E0D\u4F1A\u518D\u6B21\u8C03\u7528\u6A21\u578B\u3002\n"
            "- \u2463 \u7684\u89C4\u5219\u548C \u2464 \u7684\u62A5\u544A\u90FD\u5728\u672C\u673A\u5B9E\u65F6\u8BA1\u7B97\u3002\n"
            "- \u53EF\u4EE5\u56DE\u770B\u89C6\u9891\u7247\u6BB5\u3001\u66F4\u6B63\u8BC6\u522B\u3001\u67E5\u770B\u6BCF\u6761\u4F9D\u636E\u7684\u539F\u6587\u6458\u5F55\u3002"
        )

ui.hero()
if demo_case:
    st.markdown(
        '<div class="demo-banner"><b>\u793A\u4F8B\u6A21\u5F0F \u00B7 ' + demo_case["title"] + "</b><br>"
        + demo_case["tagline"]
        + "\u3002\u89C6\u9891\u548C\u573A\u5730\u7167\u7247\u7684\u8BC6\u522B\u7ED3\u679C\u3001\u8BF4\u6CD5\u6838\u67E5\u7ED3\u679C\u662F\u9884\u5148\u4FDD\u5B58\u7684\u771F\u5B9E\u6A21\u578B\u8FD0\u884C\u7ED3\u679C\uFF08"
        + "\u3001".join(demo_case["recorded_with"])
        + "\uFF09\uFF0C\u672A\u91CD\u65B0\u8C03\u7528\u6A21\u578B\uFF1B\u2463 \u7684\u89C4\u5219\u68C0\u67E5\u548C \u2464 \u7684\u62A5\u544A\u5728\u672C\u673A\u5B9E\u65F6\u8BA1\u7B97\u3002</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="hint">\U0001f3af <b>\u7B2C\u4E00\u6B21\u6765\uFF1F</b>\u70B9\u4EFB\u4E00\u793A\u4F8B\u6848\u4F8B\uFF0C\u4E00\u952E\u8F7D\u5165\u89C6\u9891\u3001\u573A\u5730\u7167\u7247\u548C\u5DF2\u4FDD\u5B58\u7684\u771F\u5B9E\u5206\u6790\u7ED3\u679C\uFF0C\u4E0D\u9700\u8981 API Key\u3002</div>', unsafe_allow_html=True)
    columns = st.columns(len(demo_cases)) if demo_cases else []
    for column, item in zip(columns, demo_cases):
        with column:
            ui.case_card(item["title"], item["tagline"])
            if st.button("\u8F7D\u5165\u8FD9\u4E2A\u6848\u4F8B", key=f"main-load-{item['id']}", type="primary", width="stretch"):
                load_demo_case(item["id"])
                st.rerun()

ui.step_tracker(
    [
        ("1", "\u89C6\u9891\u63D0\u53D6", bool(st.session_state.video_run), "step-1"),
        ("2", "\u8BF4\u6CD5\u6838\u67E5", bool(st.session_state.claim_check_run), "step-2"),
        ("3", "\u573A\u5730\u7167\u7247", bool(st.session_state.scene_run), "step-3"),
        ("4", "\u51C6\u5907\u68C0\u67E5", bool(st.session_state.preparation_run), "step-4"),
        ("5", "\u9650\u5B9A\u8303\u56F4\u62A5\u544A", bool(st.session_state.report_run), "step-5"),
    ]
)
st.warning("\u5404\u9636\u6BB5\u53EA\u5904\u7406\u5DF2\u8BC6\u522B\u7684\u5177\u4F53\u8BF4\u6CD5\u548C\u5DF2\u586B\u5199\u7684\u51C6\u5907\u6761\u4EF6\uFF0C\u4E0D\u8868\u793A\u6574\u6BB5\u89C6\u9891\u6B63\u786E\u3001\u4E2A\u4EBA\u9002\u7EC3\uFF0C\u6216\u5668\u6750\u548C\u573A\u5730\u5DF2\u7ECF\u901A\u8FC7\u5B89\u5168\u8BA4\u8BC1\u3002")

st.header("\u2460 \u89C6\u9891\u5185\u5BB9\u63D0\u53D6", anchor="step-1")
if demo_case:
    st.video(st.session_state.video_bytes, start_time=st.session_state.playback_start, format="video/mp4")
    if st.session_state.playback_start:
        st.caption(f"\u64AD\u653E\u5668\u5DF2\u8DF3\u5230 {format_time(st.session_state.playback_start)}\u3002\u64AD\u653E\u7531\u6D4F\u89C8\u5668\u63A7\u5236\u3002")
else:
    st.info("\u70B9\u51FB\u201C\u63D0\u53D6\u201D\u6309\u94AE\u540E\uFF0C\u5BF9\u5E94\u89C6\u9891\u6216\u56FE\u7247\u4F1A\u53D1\u9001\u5230\u672C\u673A\u914D\u7F6E\u7684\u6A21\u578B\u670D\u52A1\u3002\u8BF7\u5148\u786E\u8BA4\u7D20\u6750\u5141\u8BB8\u4E0A\u4F20\u3002")
    video_upload = st.file_uploader("\u4E0A\u4F20\u4E00\u4E2A MP4\uFF0C\u65F6\u957F 0\u2014120 \u79D2\uFF0C\u539F\u6587\u4EF6\u4E0D\u8D85\u8FC7 100MB", type=["mp4"], key="video-upload")
    sync_video(video_upload, settings)
    if video_upload is not None:
        st.video(st.session_state.video_bytes, start_time=st.session_state.playback_start, format="video/mp4")
        if st.session_state.playback_start:
            st.caption(f"\u64AD\u653E\u5668\u5DF2\u8DF3\u5230 {format_time(st.session_state.playback_start)}\u3002\u64AD\u653E\u7531\u6D4F\u89C8\u5668\u63A7\u5236\u3002")
        if st.session_state.video_validation_error:
            st.error("\u89C6\u9891\u6821\u9A8C\u5931\u8D25\uFF1A" + st.session_state.video_validation_error)
        elif st.session_state.video_metadata:
            metadata = st.session_state.video_metadata
            st.caption(
                f"\u5DF2\u5B8C\u6574\u89E3\u7801\u6821\u9A8C\uFF1A{metadata.duration_seconds:.2f} \u79D2\uFF0C{metadata.width}\u00D7{metadata.height}\uFF0C"
                f"\u89C6\u9891 {metadata.video_codec}\uFF0C\u97F3\u8F68 {metadata.audio_codec or '\u65E0'}\uFF1B\u6821\u9A8C\u8017\u65F6 {metadata.decode_seconds:.2f} \u79D2\u3002"
            )
            if metadata.audio_status == "absent":
                st.warning("\u6B64\u89C6\u9891\u6CA1\u6709\u97F3\u8F68\uFF1A\u53EA\u80FD\u63D0\u53D6\u753B\u9762\u548C\u53EF\u89C1\u5B57\u5E55\uFF0C\u4E0D\u4F1A\u751F\u6210\u201C\u542C\u5230\u7684\u539F\u8BDD\u201D\u3002")

        can_analyze_video = bool(st.session_state.video_metadata and settings.model_ready and settings.ffmpeg_path)
        if not settings.model_ready:
            st.caption("\u4E0A\u4F20\u81EA\u5DF1\u7684\u89C6\u9891\u9700\u8981\u5148\u914D\u7F6E API Key\uFF08\u89C1 README\uFF09\uFF1B\u4E5F\u53EF\u4EE5\u5728\u5DE6\u4FA7\u8F7D\u5165\u793A\u4F8B\u6848\u4F8B\u3002")
        if st.button("\u63D0\u53D6\u89C6\u9891\u5185\u5BB9", type="primary", disabled=not can_analyze_video):
            st.session_state.video_run = None
            st.session_state.video_run_error = None
            clear_claim_state()
            clear_preparation_state(clear_matches=True)
            progress_bar = st.progress(0, text="\u51C6\u5907\u5A92\u4F53")
            with st.status("\u6B63\u5728\u5904\u7406\u89C6\u9891", expanded=True) as status:
                try:
                    processing_dir = st.session_state.session_dir / "video-processing"
                    if processing_dir.exists():
                        remove_owned(processing_dir, st.session_state.session_dir)
                    started = time.perf_counter()
                    prepared = prepare_video(
                        Path(st.session_state.video_path),
                        st.session_state.video_metadata,
                        processing_dir,
                        settings.ffmpeg_path,
                        lambda message: status.write(message),
                    )
                    status.write(
                        f"\u5A92\u4F53\u65B9\u6848\uFF1A{prepared.strategy}\uFF1B{len(prepared.parts)} \u4E2A\u8BF7\u6C42\u7247\u6BB5\uFF1B"
                        f"\u5904\u7406\u8017\u65F6 {prepared.processing_seconds:.2f} \u79D2\u3002"
                    )
                    progress_bar.progress(35, text="\u8C03\u7528\u6A21\u578B\u5E76\u6821\u9A8C\u7ED3\u6784")
                    run = analyze_video(
                        prepared,
                        st.session_state.video_metadata,
                        settings,
                        revision=st.session_state.video_revision,
                        video_id=st.session_state.video_hash[:16],
                        progress=lambda message: status.write(message),
                    )
                    run_data = run_to_serializable(run)
                    run_data["media"] = {
                        "strategy": prepared.strategy,
                        "part_count": len(prepared.parts),
                        "source_size_bytes": prepared.source_size_bytes,
                        "processed_size_bytes": prepared.processed_size_bytes,
                        "processing_seconds": prepared.processing_seconds,
                    }
                    st.session_state.video_run = run_data
                    if run.result.status == "failed":
                        progress_bar.progress(100, text="\u5931\u8D25")
                        status.update(label=f"\u89C6\u9891\u63D0\u53D6\u5931\u8D25\uFF0C\u603B\u8017\u65F6 {time.perf_counter() - started:.2f} \u79D2", state="error")
                    else:
                        progress_bar.progress(100, text="\u5B8C\u6210")
                        status.update(label=f"\u89C6\u9891\u5904\u7406\u5B8C\u6210\uFF0C\u603B\u8017\u65F6 {time.perf_counter() - started:.2f} \u79D2", state="complete")
                except (MediaError, AnalysisError) as exc:
                    elapsed = time.perf_counter() - started
                    st.session_state.video_run_error = f"{exc}\uFF08\u5931\u8D25\u524D\u8017\u65F6 {elapsed:.2f} \u79D2\uFF09"
                    progress_bar.progress(100, text="\u5931\u8D25")
                    status.update(label=f"\u89C6\u9891\u5904\u7406\u5931\u8D25\uFF0C\u8017\u65F6 {elapsed:.2f} \u79D2", state="error")
                    status.write(str(exc))
if st.session_state.video_run_error:
    st.error("\u89C6\u9891\u63D0\u53D6\u5931\u8D25\uFF1A" + st.session_state.video_run_error)
if st.session_state.video_run:
    show_video_result(st.session_state.video_run)

    st.divider()
    st.header("\u2461 \u56DE\u770B\u5E76\u6838\u67E5\u5177\u4F53\u8BF4\u6CD5", anchor="step-2")
    video_result = st.session_state.video_run["result"]
    show_claim_review_editor(video_result)
    try:
        claim_candidates = build_claim_candidates(video_result, st.session_state.claim_reviews)
    except AnalysisError as exc:
        claim_candidates = []
        st.error("\u65E0\u6CD5\u51C6\u5907\u4E3B\u5F20\u5019\u9009\uFF1A" + str(exc))
    st.caption(
        f"\u5C06\u5BF9\u7167\u5F53\u524D {len(claim_candidates)} \u4E2A\u8BF4\u6CD5/\u6CE8\u610F\u4E8B\u9879\u5019\u9009\u548C\u672C\u5730\u8D44\u6599\u5361\uFF1B"
        "\u4E0D\u4F1A\u53D1\u9001\u89C6\u9891\u6587\u4EF6\u3001\u8EAB\u4F53\u6761\u4EF6\u3001\u6837\u4F8B\u6587\u5B57\u7A3F\u6216\u9884\u671F\u7B54\u6848\u3002"
    )
    can_check_claims = bool(
        video_result["status"] != "failed"
        and claim_candidates
        and knowledge
        and settings.model_ready
    )
    if demo_case and not st.session_state.claim_check_run:
        if st.session_state.claim_reviews:
            st.info("\u4F60\u66F4\u6B63\u4E86\u8BC6\u522B\u5185\u5BB9\uFF0C\u65E7\u6838\u67E5\u7ED3\u679C\u5DF2\u5931\u6548\u3002\u91CD\u65B0\u6838\u67E5\u9700\u8981 API Key\uFF1B\u4E5F\u53EF\u4EE5\u64A4\u9500\u66F4\u6B63\u540E\u6062\u590D\u793A\u4F8B\u7ED3\u679C\u3002")
        elif st.button("\u6062\u590D\u793A\u4F8B\u6838\u67E5\u7ED3\u679C", type="primary"):
            st.session_state.claim_check_run = demo_case["claim_run"]
            clear_report_state()
            st.rerun()
    if st.button("\u6839\u636E\u672C\u5730\u8D44\u6599\u6838\u67E5\u89C6\u9891\u8BF4\u6CD5", type="primary", disabled=not can_check_claims):
        st.session_state.claim_check_run = None
        st.session_state.claim_check_error = None
        st.session_state.claim_check_failure = None
        clear_report_state()
        st.session_state.claim_check_revision += 1
        started = time.perf_counter()
        progress_bar = st.progress(0, text="\u52A0\u8F7D\u672C\u5730\u8D44\u6599")
        with st.status("\u6B63\u5728\u6838\u67E5\u5177\u4F53\u8BF4\u6CD5", expanded=True) as status:
            try:
                progress_bar.progress(30, text="\u8C03\u7528\u6A21\u578B\u8FDB\u884C\u8D44\u6599\u5BF9\u7167")
                run = analyze_claims(
                    video_result,
                    st.session_state.claim_reviews,
                    knowledge,
                    settings,
                    revision=st.session_state.claim_check_revision,
                    progress=lambda message: status.write(message),
                )
                st.session_state.claim_check_run = claim_check_to_serializable(run)
                progress_bar.progress(100, text="\u5B8C\u6210")
                status.update(label=f"\u4E3B\u5F20\u6838\u67E5\u5B8C\u6210\uFF0C\u603B\u8017\u65F6 {time.perf_counter() - started:.2f} \u79D2", state="complete")
            except AnalysisError as exc:
                elapsed = time.perf_counter() - started
                st.session_state.claim_check_error = f"{exc}\uFF08\u5931\u8D25\u524D\u8017\u65F6 {elapsed:.2f} \u79D2\uFF09"
                if isinstance(exc, ClaimCheckError):
                    st.session_state.claim_check_failure = exc.details()
                progress_bar.progress(100, text="\u5931\u8D25")
                status.update(label=f"\u4E3B\u5F20\u6838\u67E5\u5931\u8D25\uFF0C\u8017\u65F6 {elapsed:.2f} \u79D2", state="error")
                status.write(str(exc))
    if st.session_state.claim_check_error:
        st.error("\u4E3B\u5F20\u6838\u67E5\u5931\u8D25\uFF1A" + st.session_state.claim_check_error)
        if st.session_state.claim_check_failure:
            request_stats(st.session_state.claim_check_failure)
            with st.expander("\u67E5\u770B\u5931\u8D25\u7684\u6A21\u578B\u539F\u59CB\u56DE\u590D"):
                raw = st.session_state.claim_check_failure["raw_responses"][0]
                st.error(raw.get("parse_error", "\u6A21\u578B\u56DE\u590D\u672A\u901A\u8FC7\u6821\u9A8C"))
                st.code(raw["text"], language="json")
    if st.session_state.claim_check_run and knowledge:
        show_claim_check_result(st.session_state.claim_check_run, video_result, knowledge)

st.divider()
st.header("\u2462 \u573A\u5730\u7167\u7247", anchor="step-3")
st.caption("\u8FD9\u91CC\u63D0\u53D6\u573A\u5730\u5019\u9009\u4E8B\u5B9E\uFF1B\u7A33\u5B9A\u3001\u9632\u6ED1\u3001\u627F\u91CD\u548C\u662F\u5426\u4E3A\u672C\u6B21\u5B9E\u9645\u5668\u6750\u4ECD\u987B\u5728\u4E0B\u4E00\u6B65\u7531\u672C\u4EBA\u786E\u8BA4\u3002")
if demo_case:
    if not demo_case["scenes"]:
        st.info("\u8FD9\u4E2A\u6848\u4F8B\u805A\u7126\u89C6\u9891\u8BF4\u6CD5\u6838\u67E5\uFF0C\u6CA1\u6709\u573A\u5730\u7167\u7247\u3002\u2463 \u4ECD\u53EF\u68C0\u67E5\u672C\u4EBA\u6761\u4EF6\u3002")
    else:
        scene_ids = [item["id"] for item in demo_case["scenes"]]
        if len(scene_ids) > 1:
            labels = {item["id"]: item["label"] for item in demo_case["scenes"]}
            choice = st.radio(
                "\u5207\u6362\u573A\u5730\u7167\u7247",
                scene_ids,
                index=scene_ids.index(st.session_state.demo_scene_id) if st.session_state.demo_scene_id in scene_ids else 0,
                format_func=lambda value: labels[value],
                horizontal=True,
            )
            if choice != st.session_state.demo_scene_id:
                install_demo_scene(demo_case, choice)
                st.rerun()
        columns = st.columns(len(st.session_state.demo_images))
        for index, (column, data) in enumerate(zip(columns, st.session_state.demo_images), start=1):
            column.image(data, caption=f"photo-{index}", width="stretch")
else:
    image_uploads = st.file_uploader(
        "\u4E0A\u4F20 2\u20144 \u5F20 JPG/PNG\uFF0C\u6BCF\u5F20\u4E0D\u8D85\u8FC7 5MB",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="image-upload",
    )
    sync_images(image_uploads)
    if image_uploads:
        if not 2 <= len(image_uploads) <= 4:
            st.session_state.image_validation_error = "\u5FC5\u987B\u4E0A\u4F20 2\u20144 \u5F20\u56FE\u7247"
        columns = st.columns(min(len(image_uploads), 4))
        for index, (column, uploaded) in enumerate(zip(columns, image_uploads), start=1):
            column.image(uploaded.getvalue(), caption=f"photo-{index}", width="stretch")
        if st.session_state.image_validation_error:
            st.error("\u56FE\u7247\u6821\u9A8C\u5931\u8D25\uFF1A" + st.session_state.image_validation_error)
        elif len(st.session_state.image_metadata) == len(image_uploads):
            summary = "\uFF1B".join(
                f"photo-{index} {item.format} {item.width}\u00D7{item.height}"
                for index, item in enumerate(st.session_state.image_metadata, start=1)
            )
            st.caption(f"\u5DF2\u5B9E\u9645\u89E3\u7801\uFF1A{summary}\uFF1B\u6821\u9A8C\u8017\u65F6 {st.session_state.image_validation_seconds:.2f} \u79D2\u3002")

        can_analyze_scene = bool(
            2 <= len(image_uploads) <= 4
            and not st.session_state.image_validation_error
            and len(st.session_state.image_metadata) == len(image_uploads)
            and settings.model_ready
        )
        if st.button("\u63D0\u53D6\u573A\u5730\u4FE1\u606F", type="primary", disabled=not can_analyze_scene):
            st.session_state.scene_run = None
            st.session_state.scene_run_error = None
            clear_preparation_state(clear_area=True)
            progress_bar = st.progress(0, text="\u51C6\u5907\u56FE\u7247")
            with st.status("\u6B63\u5728\u5904\u7406\u573A\u5730\u56FE\u7247", expanded=True) as status:
                try:
                    started = time.perf_counter()
                    progress_bar.progress(30, text="\u8C03\u7528\u6A21\u578B\u5E76\u6821\u9A8C\u7ED3\u6784")
                    run = analyze_scene(
                        st.session_state.image_metadata,
                        settings,
                        revision=st.session_state.scene_revision,
                        progress=lambda message: status.write(message),
                    )
                    st.session_state.scene_run = run_to_serializable(run)
                    if run.result.status == "failed":
                        progress_bar.progress(100, text="\u5931\u8D25")
                        status.update(label=f"\u573A\u5730\u63D0\u53D6\u5931\u8D25\uFF0C\u603B\u8017\u65F6 {time.perf_counter() - started:.2f} \u79D2", state="error")
                    else:
                        progress_bar.progress(100, text="\u5B8C\u6210")
                        status.update(label=f"\u573A\u5730\u5904\u7406\u5B8C\u6210\uFF0C\u603B\u8017\u65F6 {time.perf_counter() - started:.2f} \u79D2", state="complete")
                except AnalysisError as exc:
                    elapsed = time.perf_counter() - started
                    st.session_state.scene_run_error = f"{exc}\uFF08\u5931\u8D25\u524D\u8017\u65F6 {elapsed:.2f} \u79D2\uFF09"
                    progress_bar.progress(100, text="\u5931\u8D25")
                    status.update(label=f"\u573A\u5730\u5904\u7406\u5931\u8D25\uFF0C\u8017\u65F6 {elapsed:.2f} \u79D2", state="error")
                    status.write(str(exc))
if st.session_state.scene_run_error:
    st.error("\u573A\u5730\u63D0\u53D6\u5931\u8D25\uFF1A" + st.session_state.scene_run_error)
if st.session_state.scene_run:
    show_scene_result(st.session_state.scene_run)

st.divider()
st.header("\u2463 \u68C0\u67E5\u672C\u4EBA\u3001\u573A\u5730\u548C\u5668\u6750\u51C6\u5907", anchor="step-4")
if not st.session_state.video_run:
    st.info("\u5148\u5B8C\u6210\u89C6\u9891\u5185\u5BB9\u63D0\u53D6\uFF1B\u51C6\u5907\u89C4\u5219\u9700\u8981\u5F53\u524D\u89C6\u9891\u4E2D\u5DF2\u8BC6\u522B\u7684\u52A8\u4F5C\u3002")
else:
    preparation_video_result = st.session_state.video_run["result"]
    preparation_scene_result = st.session_state.scene_run["result"] if st.session_state.scene_run else None
    if preparation_video_result["status"] == "failed":
        st.error("\u89C6\u9891\u63D0\u53D6\u5931\u8D25\uFF0C\u4E0D\u80FD\u4F9D\u636E\u8BE5\u89C6\u9891\u7684\u52A8\u4F5C\u8FD0\u884C\u51C6\u5907\u89C4\u5219\u3002")
    else:
        if demo_case and knowledge:
            presets = build_presets(demo_case)
            if presets:
                st.markdown("**\u4E00\u952E\u5957\u7528\u793A\u4F8B\u6761\u4EF6**\uFF08\u672C\u4EBA\u6761\u4EF6\u4E3A\u865A\u6784\uFF0C\u89C4\u5219\u5728\u672C\u673A\u5B9E\u65F6\u8BA1\u7B97\uFF0C\u4E0D\u8C03\u7528\u6A21\u578B\uFF09")
                columns = st.columns(len(presets))
                for column, preset in zip(columns, presets):
                    with column:
                        if st.button(f"{preset.preset_id} \u00B7 {preset.label}", key=f"preset-{preset.preset_id}", help=preset.description, width="stretch"):
                            apply_preset(demo_case, preset)
                            st.rerun()
                        st.caption(preset.description)
        show_preparation_form(preparation_video_result, preparation_scene_result, knowledge)
        if st.session_state.preparation_error:
            st.error("\u51C6\u5907\u68C0\u67E5\u5931\u8D25\uFF1A" + st.session_state.preparation_error)
        if st.session_state.preparation_run and knowledge:
            show_preparation_result(st.session_state.preparation_run, preparation_video_result, knowledge)

st.divider()
st.header("\u2464 \u751F\u6210\u672C\u6B21\u9650\u5B9A\u8303\u56F4\u62A5\u544A", anchor="step-5")
st.caption(
    "\u62A5\u544A\u53EA\u6C47\u603B\u5F53\u524D\u4F1A\u8BDD\u4E2D\u5DF2\u7ECF\u5F62\u6210\u7684\u7ED3\u679C\uFF0C\u4E0D\u91CD\u65B0\u8C03\u7528\u6A21\u578B\u3002\u4E3B\u5F20\u6216\u51C6\u5907\u9636\u6BB5\u672A\u5B8C\u6210\u65F6\u4ECD\u53EF\u751F\u6210\uFF0C"
    "\u4F46\u4F1A\u660E\u786E\u6807\u4E3A partial/failed\uFF0C\u4E0D\u80FD\u663E\u793A\u6210\u6574\u4F53\u901A\u8FC7\u3002"
)
can_generate_report = bool(st.session_state.video_run and knowledge)
if not st.session_state.video_run:
    st.info("\u81F3\u5C11\u9700\u8981\u4E00\u6B21\u89C6\u9891\u63D0\u53D6\u7ED3\u679C\u624D\u80FD\u751F\u6210\u62A5\u544A\uFF1B\u89C6\u9891\u5931\u8D25\u4E5F\u4F1A\u5982\u5B9E\u8BB0\u5F55\u4E3A\u5931\u8D25\u62A5\u544A\u3002")
if st.button("\u751F\u6210\u6216\u66F4\u65B0\u9650\u5B9A\u8303\u56F4\u62A5\u544A\uFF08\u4EC5\u672C\u5730\uFF09", type="primary", disabled=not can_generate_report):
    next_report_revision = st.session_state.report_revision + 1
    st.session_state.report_run = None
    st.session_state.report_error = None
    try:
        report_run = build_report(
            st.session_state.video_run,
            st.session_state.scene_run,
            st.session_state.claim_check_run,
            st.session_state.preparation_run,
            knowledge,
            revision=next_report_revision,
            claim_error=st.session_state.claim_check_error,
            preparation_error=st.session_state.preparation_error,
        )
        st.session_state.report_revision = next_report_revision
        st.session_state.report_run = report_to_serializable(report_run)
    except AnalysisError as exc:
        st.session_state.report_error = str(exc)
if st.session_state.report_error:
    st.error("\u62A5\u544A\u751F\u6210\u5931\u8D25\uFF1A" + st.session_state.report_error)
if st.session_state.report_run:
    show_formal_report(st.session_state.report_run)
