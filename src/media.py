from __future__ import annotations

import base64
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError


MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 120.0
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_ENCODED_BYTES = 9_500_000
TARGET_TRANSCODE_BYTES = 6_500_000


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    size_bytes: int
    container: str
    video_codec: str
    width: int
    height: int
    audio_status: str
    audio_codec: str | None
    decode_seconds: float


@dataclass(frozen=True)
class MediaPart:
    path: Path
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class PreparedVideo:
    strategy: str
    parts: list[MediaPart]
    source_size_bytes: int
    processed_size_bytes: int
    processing_seconds: float


@dataclass(frozen=True)
class ImageMetadata:
    path: Path
    size_bytes: int
    width: int
    height: int
    format: str
    mime_type: str


def encoded_data_uri_size(path: Path) -> int:
    return 13 + 4 * math.ceil(path.stat().st_size / 3)


def _run(args: list[str], timeout: float = 240) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError("媒体工具运行超时") from exc
    except OSError as exc:
        raise MediaError(f"无法运行媒体工具：{exc}") from exc


def validate_video(path: Path, original_name: str, ffmpeg_path: str | None) -> VideoMetadata:
    if not ffmpeg_path:
        raise MediaError("未找到 FFmpeg。请安装后加入 PATH，或设置 FFMPEG_PATH。")
    if Path(original_name).suffix.lower() != ".mp4":
        raise MediaError("只接受扩展名为 .mp4 的视频")
    size = path.stat().st_size
    if size <= 0:
        raise MediaError("视频文件为空")
    if size > MAX_VIDEO_BYTES:
        raise MediaError("视频原文件超过 100MB")

    metadata_run = _run([ffmpeg_path, "-hide_banner", "-i", str(path)], timeout=30)
    details = f"{metadata_run.stdout}\n{metadata_run.stderr}"
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)
    input_match = re.search(r"Input #0,\s*(.+?),\s+from\s", details)
    video_match = re.search(r"Video:\s*([^,\s]+).*?,\s*(\d{2,5})x(\d{2,5})", details)
    audio_match = re.search(r"Audio:\s*([^,\s]+)", details)
    if not duration_match or not input_match or not video_match:
        raise MediaError("文件不是可识别且含视频流的 MP4")

    container = input_match.group(1)
    if "mp4" not in container.lower() and "mov" not in container.lower():
        raise MediaError(f"实际容器不是 MP4（检测到 {container}）")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        raise MediaError("视频时长必须大于 0 秒")
    if duration > MAX_VIDEO_SECONDS + 0.05:
        raise MediaError(f"视频时长 {duration:.2f} 秒，超过 120 秒")

    decode_started = time.perf_counter()
    decode_run = _run(
        [ffmpeg_path, "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"],
        timeout=max(60, duration * 3),
    )
    decode_seconds = time.perf_counter() - decode_started
    if decode_run.returncode != 0:
        error_tail = (decode_run.stderr or "未知解码错误")[-600:].strip()
        raise MediaError(f"视频完整解码失败：{error_tail}")

    return VideoMetadata(
        duration_seconds=duration,
        size_bytes=size,
        container=container,
        video_codec=video_match.group(1),
        width=int(video_match.group(2)),
        height=int(video_match.group(3)),
        audio_status="present" if audio_match else "absent",
        audio_codec=audio_match.group(1) if audio_match else None,
        decode_seconds=decode_seconds,
    )


def _transcode(
    source: Path,
    target: Path,
    ffmpeg_path: str,
    duration: float,
    has_audio: bool,
    start_seconds: float | None = None,
) -> None:
    audio_kbps = 64 if has_audio else 0
    target_kbps = max(220, int(TARGET_TRANSCODE_BYTES * 8 / max(duration, 1) / 1000))
    video_kbps = max(160, min(1200, target_kbps - audio_kbps - 24))
    args = [ffmpeg_path, "-y"]
    if start_seconds is not None:
        args += ["-ss", f"{start_seconds:.3f}", "-t", f"{duration:.3f}"]
    args += ["-i", str(source), "-map", "0:v:0"]
    if has_audio:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", f"{audio_kbps}k"]
    else:
        args += ["-an"]
    args += [
        "-vf",
        "scale=960:-2:force_original_aspect_ratio=decrease",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{video_kbps}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    result = _run(args, timeout=max(120, duration * 5))
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        error_tail = (result.stderr or "未知转码错误")[-800:].strip()
        raise MediaError(f"视频转码失败：{error_tail}")


def prepare_video(
    source: Path,
    metadata: VideoMetadata,
    work_dir: Path,
    ffmpeg_path: str,
    progress: Callable[[str], None] | None = None,
) -> PreparedVideo:
    started = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    if encoded_data_uri_size(source) < MAX_ENCODED_BYTES:
        return PreparedVideo(
            strategy="original",
            parts=[MediaPart(source, 0.0, metadata.duration_seconds)],
            source_size_bytes=source.stat().st_size,
            processed_size_bytes=source.stat().st_size,
            processing_seconds=time.perf_counter() - started,
        )

    if progress:
        progress("原视频 Base64 超过接口限制，正在保留声音压缩整段视频")
    transcoded = work_dir / "whole.mp4"
    _transcode(
        source,
        transcoded,
        ffmpeg_path,
        metadata.duration_seconds,
        metadata.audio_status == "present",
    )
    checked = validate_video(transcoded, "whole.mp4", ffmpeg_path)
    if metadata.audio_status == "present" and checked.audio_status != "present":
        raise MediaError("压缩后音轨丢失，已停止分析")
    if abs(checked.duration_seconds - metadata.duration_seconds) > max(0.7, metadata.duration_seconds * 0.02):
        raise MediaError("压缩后视频时长发生异常变化，已停止分析")
    if encoded_data_uri_size(transcoded) < MAX_ENCODED_BYTES:
        return PreparedVideo(
            strategy="transcoded_whole",
            parts=[MediaPart(transcoded, 0.0, checked.duration_seconds)],
            source_size_bytes=source.stat().st_size,
            processed_size_bytes=transcoded.stat().st_size,
            processing_seconds=time.perf_counter() - started,
        )

    if progress:
        progress("整段压缩后仍超过接口限制，正在按实际大小分段")
    safe_raw_bytes = int((MAX_ENCODED_BYTES - 13) * 3 / 4)
    seconds_per_part = max(
        12.0,
        min(metadata.duration_seconds, checked.duration_seconds * safe_raw_bytes / transcoded.stat().st_size * 0.82),
    )
    parts: list[MediaPart] = []
    start = 0.0
    index = 1
    while start < metadata.duration_seconds - 0.05:
        part_duration = min(seconds_per_part, metadata.duration_seconds - start)
        target = work_dir / f"part-{index:02d}.mp4"
        _transcode(
            source,
            target,
            ffmpeg_path,
            part_duration,
            metadata.audio_status == "present",
            start_seconds=start,
        )
        part_checked = validate_video(target, target.name, ffmpeg_path)
        if metadata.audio_status == "present" and part_checked.audio_status != "present":
            raise MediaError(f"第 {index} 段音轨丢失")
        if encoded_data_uri_size(target) >= MAX_ENCODED_BYTES:
            raise MediaError(f"第 {index} 段仍超过接口 Base64 限制")
        parts.append(MediaPart(target, start, min(part_checked.duration_seconds, part_duration)))
        start += part_duration
        index += 1

    return PreparedVideo(
        strategy="segmented",
        parts=parts,
        source_size_bytes=source.stat().st_size,
        processed_size_bytes=sum(part.path.stat().st_size for part in parts),
        processing_seconds=time.perf_counter() - started,
    )


def validate_image(path: Path) -> ImageMetadata:
    size = path.stat().st_size
    if size <= 0:
        raise MediaError("图片文件为空")
    if size > MAX_IMAGE_BYTES:
        raise MediaError("单张图片超过 5MB")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            detected_format = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaError("图片损坏或不是可解码的图片") from exc
    if detected_format not in {"JPEG", "PNG"}:
        raise MediaError(f"只接受实际格式为 JPG/PNG 的图片（检测到 {detected_format or '未知'}）")
    if width <= 0 or height <= 0:
        raise MediaError("图片尺寸无效")
    mime = "image/jpeg" if detected_format == "JPEG" else "image/png"
    return ImageMetadata(path, size, width, height, detected_format, mime)


def to_data_uri(path: Path, mime_type: str = "") -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

