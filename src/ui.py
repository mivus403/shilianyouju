"""页面样式与小型展示组件。只生成 HTML 片段，不含业务逻辑。"""
from __future__ import annotations

import html

import streamlit as st

CLAIM_STATUS = {
    "supported_limited": ("有限范围有依据", "#2e9e6b"),
    "conflicts": ("与资料要求冲突", "#d64545"),
    "overstated": ("表述超出依据", "#e08a1e"),
    "insufficient": ("当前资料不足", "#6b7f8f"),
    "recognition_needed": ("识别待确认", "#7a5af8"),
}
PREPARATION_STATUS = {
    "condition_gap": ("条件缺口", "#d64545"),
    "needs_confirmation": ("需要确认", "#e08a1e"),
    "unsupported": ("超出支持范围", "#6b7f8f"),
}

CSS = """
<style>
.block-container {max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem;}
h1, h2, h3 {letter-spacing: .01em;}
h2 {border-left: 6px solid #1fa096; padding-left: .6rem; margin-top: .4rem;}
[data-testid="stSidebar"] {background: #eaf3f1;}
[data-testid="stSidebar"] .stButton button {width: 100%; text-align: left; justify-content: flex-start;}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px; background: #ffffff; border: 1px solid #dbe7e4;
  box-shadow: 0 1px 3px rgba(18,60,76,.06);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-status) {border-left-width: 6px;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-supported_limited) {border-left-color: #2e9e6b;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-conflicts), div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-condition_gap) {border-left-color: #d64545;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-overstated), div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-needs_confirmation) {border-left-color: #e08a1e;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-insufficient), div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-unsupported) {border-left-color: #6b7f8f;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.s-recognition_needed) {border-left-color: #7a5af8;}
.card-status {display: none;}
.hero {
  background: linear-gradient(135deg, #123c4c 0%, #1b6b6b 60%, #1fa096 100%);
  color: #fff; border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(18,60,76,.22);
}
.hero h1 {font-size: 2.6rem; margin: 0 0 .2rem 0; color: #fff; padding: 0;}
.hero .sub {font-size: 1.15rem; color: #d3f1ec; margin-bottom: 1rem;}
.hero .pills {display: flex; flex-wrap: wrap; gap: .6rem;}
.hero .pill {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); border-radius: 999px; padding: .35rem .9rem; font-size: .95rem;}
.steps {display: flex; gap: .5rem; flex-wrap: wrap; margin: .2rem 0 1rem 0;}
.steps a {
  flex: 1 1 150px; text-decoration: none; border-radius: 12px; padding: .55rem .8rem;
  border: 1px solid #cfe0dd; background: #fff; color: #17323b; font-size: .95rem;
  transition: transform .12s ease, box-shadow .12s ease;
}
.steps a:hover {transform: translateY(-2px); box-shadow: 0 4px 10px rgba(18,60,76,.12);}
.steps a .n {display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; text-align: center; border-radius: 50%; background: #cfe0dd; margin-right: .45rem; font-weight: 700;}
.steps a.done {border-color: #2e9e6b; background: #eef9f3;}
.steps a.done .n {background: #2e9e6b; color: #fff;}
.steps a.todo {color: #6b7f8f;}
.chip {display: inline-block; border-radius: 999px; padding: .12rem .75rem; font-size: .85rem; font-weight: 600; color: #fff; margin-right: .4rem;}
.chiprow {display: flex; flex-wrap: wrap; gap: .5rem; margin: .3rem 0 .8rem 0;}
.stat {border-radius: 14px; padding: .6rem 1rem; background: #fff; border: 1px solid #dbe7e4; min-width: 130px;}
.stat b {font-size: 1.6rem; display: block; line-height: 1.2;}
.stat span {font-size: .85rem; color: #51666e;}
.demo-banner {border-radius: 14px; padding: .8rem 1.1rem; background: #fff7e6; border: 1px solid #f2d089; margin-bottom: .8rem;}
.case-card {border-radius: 16px; padding: 1rem 1.1rem; background: #fff; border: 1px solid #dbe7e4; min-height: 13rem;}
.case-card b {font-size: 1.05rem;}
.case-card p {color: #51666e; font-size: .92rem; margin: .4rem 0 0 0;}
.hint {border-radius: 12px; padding: .6rem .9rem; background: #eaf3f1; color: #17323b; margin: .3rem 0 .8rem 0;}
</style>
"""


LABELS = {
    "clear": "识别清晰",
    "uncertain": "识别不确定",
    "unclear": "不清楚",
    "supported": "在支持范围内",
    "out_of_scope": "超出支持范围",
    "other": "超出支持范围",
    "speech": "语音",
    "caption": "字幕",
    "visual": "画面",
    "quote": "原话",
    "paraphrase": "概述",
    "endorsed": "视频直接主张",
    "rejected": "视频明确反对",
    "quoted_only": "只是转述",
}


def label(value: str | None) -> str:
    return LABELS.get(value, value or "未知") if value else "未知"


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>适练有据</h1>
  <div class="sub">跟着视频做居家运动之前，先查两件事：视频里的说法有没有依据，自己家的条件适不适合</div>
  <div class="pills">
    <span class="pill">① 逐条核查视频说法</span>
    <span class="pill">② 对照本人条件与场地</span>
    <span class="pill">③ 生成限定范围报告</span>
    <span class="pill">不做诊断 · 不开处方 · 不打安全分</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def step_tracker(steps: list[tuple[str, str, bool, str]]) -> None:
    items = []
    for number, label, done, anchor in steps:
        css = "done" if done else "todo"
        mark = "✓" if done else number
        items.append(f'<a class="{css}" href="#{anchor}"><span class="n">{mark}</span>{html.escape(label)}</a>')
    st.markdown('<div class="steps">' + "".join(items) + "</div>", unsafe_allow_html=True)


def chip(label: str, color: str) -> str:
    return f'<span class="chip" style="background:{color}">{html.escape(label)}</span>'


def status_marker(status: str) -> None:
    st.markdown(f'<span class="card-status s-{status}"></span>', unsafe_allow_html=True)


def stat_row(items: list[tuple[str, int, str]]) -> None:
    cells = "".join(
        f'<div class="stat" style="border-top:4px solid {color}"><b>{count}</b><span>{html.escape(label)}</span></div>'
        for label, count, color in items
    )
    st.markdown(f'<div class="chiprow">{cells}</div>', unsafe_allow_html=True)


def case_card(title: str, tagline: str) -> None:
    st.markdown(
        f'<div class="case-card"><b>{html.escape(title)}</b><p>{html.escape(tagline)}</p></div>',
        unsafe_allow_html=True,
    )
