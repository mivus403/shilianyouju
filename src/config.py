from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


DEFAULT_MODEL = "qwen3.8-omni-flash"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str | None
    model: str
    ffmpeg_path: str | None
    request_timeout_seconds: float

    @property
    def missing_model_config(self) -> list[str]:
        missing: list[str] = []
        if not self.api_key:
            missing.append("DASHSCOPE_API_KEY")
        if not self.base_url:
            missing.append("DASHSCOPE_BASE_URL")
        elif "{" in self.base_url or "}" in self.base_url:
            missing.append("DASHSCOPE_BASE_URL（仍含占位符）")
        elif urlparse(self.base_url).scheme != "https":
            missing.append("DASHSCOPE_BASE_URL（必须为 HTTPS）")
        return missing

    @property
    def model_ready(self) -> bool:
        return not self.missing_model_config

    @property
    def base_url_host(self) -> str | None:
        return urlparse(self.base_url).hostname if self.base_url else None


def load_settings(project_root: Path | None = None) -> Settings:
    if project_root:
        load_dotenv(project_root / ".env", override=False)
    else:
        load_dotenv(override=False)

    configured_ffmpeg = os.getenv("FFMPEG_PATH")
    ffmpeg_path: str | None = None
    if configured_ffmpeg:
        candidate = Path(configured_ffmpeg).expanduser()
        if candidate.is_file():
            ffmpeg_path = str(candidate.resolve())
    if not ffmpeg_path:
        ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg_path = None

    return Settings(
        api_key=os.getenv("DASHSCOPE_API_KEY") or None,
        base_url=(os.getenv("DASHSCOPE_BASE_URL") or "").rstrip("/") or None,
        model=os.getenv("QWEN_OMNI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        ffmpeg_path=ffmpeg_path,
        request_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "240")),
    )

