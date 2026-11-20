from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_claim_check_scope_and_uploaders():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    assert any("适练有据" in item.value for item in app.markdown)
    assert len(app.get("file_uploader")) == 2
    assert "④ 检查本人、场地和器材准备" in [header.value for header in app.header]
    assert "⑤ 生成本次限定范围报告" in [header.value for header in app.header]


def test_preparation_form_renders_from_existing_video_without_scene():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    app.session_state["video_run"] = {
        "result": {
            "schema_version": "0.1",
            "analysis_id": "video-ui-test",
            "revision": 1,
            "video_id": "video-ui-hash",
            "duration_seconds": 8.0,
            "audio_status": "present",
            "status": "complete",
            "unprocessed_ranges": [],
            "actions": [
                {
                    "id": "action-1",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "type": "sit_to_stand",
                    "description": "人物从椅子站起",
                    "equipment": ["chair"],
                    "recognition": "clear",
                    "scope": "supported",
                }
            ],
            "claims": [],
            "cautions": [],
            "uncertainties": [],
        },
        "raw_responses": [],
        "requests": [],
    }
    app.run(timeout=20)
    assert not app.exception
    submit = next(button for button in app.button if button.label == "检查准备条件（仅本地）")
    assert any(select.label == "目前是否有不适？" for select in app.selectbox)
    assert any("不会发送给模型" in caption.value for caption in app.caption)
    submit.click()
    app.run(timeout=20)
    assert not app.exception
    assert app.session_state["preparation_run"]["model_requests"] == 0
    assert app.session_state["preparation_run"]["result"]["status"] == "partial"
    report_submit = next(button for button in app.button if button.label == "生成或更新限定范围报告（仅本地）")
    report_submit.click()
    app.run(timeout=20)
    assert not app.exception
    assert app.session_state["report_run"]["result"]["workflow_status"] == "partial"
    assert app.session_state["report_run"]["result"]["model_requests_during_report"] == 0
    assert len(app.get("download_button")) == 2
    resubmit = next(button for button in app.button if button.label == "检查准备条件（仅本地）")
    resubmit.click()
    app.run(timeout=20)
    assert not app.exception
    assert app.session_state["report_run"] is None
