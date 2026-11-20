import json
import subprocess
from pathlib import Path

import pytest

import src.media as media_module
from src.config import load_settings
from src.media import MediaError, encoded_data_uri_size, prepare_video, validate_image, validate_video


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = Path(load_settings(ROOT).ffmpeg_path or "ffmpeg-not-found")


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_all_sample_videos_decode_with_audio():
    expected = {"case_a_claims": 33, "case_b_preparation": 27, "case_c_unknown": 25}
    for case, seconds in expected.items():
        path = ROOT / "samples" / case / "video.mp4"
        metadata = validate_video(path, path.name, str(FFMPEG))
        assert seconds <= metadata.duration_seconds < seconds + 1
        assert metadata.audio_status == "present"


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_corrupt_mp4_is_rejected(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(MediaError):
        validate_video(path, path.name, str(FFMPEG))


def test_sample_images_are_real_png():
    paths = list((ROOT / "samples").glob("case_*/*.png"))
    scene_paths = [path for path in paths if path.name.startswith("scene_")]
    assert len(scene_paths) == 11
    for path in scene_paths:
        metadata = validate_image(path)
        assert metadata.format == "PNG"
        assert metadata.width > 0 and metadata.height > 0


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_real_world_sample_videos_decode_with_audio_and_fit_direct_request():
    expected = {"case_d_real_seated": 14, "case_e_real_mixed": 28}
    for case, seconds in expected.items():
        path = ROOT / "samples" / "real_world" / case / "video.mp4"
        metadata = validate_video(path, path.name, str(FFMPEG))
        assert seconds <= metadata.duration_seconds < seconds + 1
        assert metadata.audio_status == "present"
        assert encoded_data_uri_size(path) < 9_500_000


def test_real_world_scene_frames_are_jpeg():
    paths = list((ROOT / "samples" / "real_world").glob("case_*/scene_frame_*.jpg"))
    assert len(paths) == 6
    for path in paths:
        metadata = validate_image(path)
        assert metadata.format == "JPEG"
        assert metadata.width == 1280 and metadata.height == 720


def test_manifest_ready_files_exist_and_stay_inside_samples():
    samples_root = (ROOT / "samples").resolve()
    manifest = json.loads((samples_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "mixed_development_fixtures"
    assert [case["id"] for case in manifest["cases"]] == ["A", "B", "C", "D", "E"]
    allowed_kinds = {"real_video", "real_photo", "synthetic_video", "synthetic_image", "script"}
    for case in manifest["cases"]:
        for item in case["files"]:
            path = (samples_root / item["local_path"]).resolve()
            assert path.is_relative_to(samples_root)
            assert item["kind"] in allowed_kinds
            assert item["status"] == "ready"
            assert path.is_file()


def test_fake_png_is_rejected(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"plain text")
    with pytest.raises(MediaError):
        validate_image(path)


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_small_sample_uses_whole_video(tmp_path):
    source = ROOT / "samples" / "case_a_claims" / "video.mp4"
    metadata = validate_video(source, source.name, str(FFMPEG))
    prepared = prepare_video(source, metadata, tmp_path / "prepared", str(FFMPEG))
    assert prepared.strategy == "original"
    assert len(prepared.parts) == 1
    assert prepared.parts[0].start_seconds == 0
    assert encoded_data_uri_size(prepared.parts[0].path) < 9_500_000


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_video_without_audio_is_reported(tmp_path):
    source = ROOT / "samples" / "case_a_claims" / "video.mp4"
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [str(FFMPEG), "-y", "-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", str(silent)],
        check=True,
        capture_output=True,
    )
    metadata = validate_video(silent, silent.name, str(FFMPEG))
    assert metadata.audio_status == "absent"


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_video_over_120_seconds_is_rejected(tmp_path):
    long_video = tmp_path / "too-long.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=1",
            "-t",
            "121",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(long_video),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(MediaError, match="超过 120 秒"):
        validate_video(long_video, long_video.name, str(FFMPEG))


@pytest.mark.skipif(not FFMPEG.is_file(), reason="本机临时 FFmpeg 不存在")
def test_adaptive_segments_keep_audio_and_original_offsets(tmp_path, monkeypatch):
    source = ROOT / "samples" / "case_a_claims" / "video.mp4"
    metadata = validate_video(source, source.name, str(FFMPEG))
    monkeypatch.setattr(media_module, "MAX_ENCODED_BYTES", 500_000)
    monkeypatch.setattr(media_module, "TARGET_TRANSCODE_BYTES", 300_000)
    prepared = prepare_video(source, metadata, tmp_path / "segmented", str(FFMPEG))
    assert prepared.strategy == "segmented"
    assert len(prepared.parts) == 2
    assert prepared.parts[0].start_seconds == 0
    assert prepared.parts[1].start_seconds > 0
    for part in prepared.parts:
        checked = validate_video(part.path, part.path.name, str(FFMPEG))
        assert checked.audio_status == "present"
        assert encoded_data_uri_size(part.path) < media_module.MAX_ENCODED_BYTES
