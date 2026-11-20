from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
import uuid
from pathlib import Path


APP_TEMP_ROOT = Path(tempfile.gettempdir()) / "shilianyouju"


def create_session_dir() -> Path:
    APP_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    target = APP_TEMP_ROOT / f"session-{uuid.uuid4().hex}"
    target.mkdir()
    return target


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_owned_file(session_dir: Path, data: bytes, name: str) -> Path:
    if not session_dir.resolve().is_relative_to(APP_TEMP_ROOT.resolve()):
        raise ValueError("临时目录不属于本应用")
    path = session_dir / name
    path.write_bytes(data)
    return path


def remove_owned(path: Path, session_dir: Path) -> None:
    resolved = path.resolve()
    owner = session_dir.resolve()
    if not resolved.is_relative_to(owner) or resolved == owner:
        raise ValueError("拒绝删除不属于当前会话的路径")
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
    elif resolved.exists():
        resolved.unlink()


def prune_stale_sessions(max_age_hours: int = 24) -> None:
    if not APP_TEMP_ROOT.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for child in APP_TEMP_ROOT.iterdir():
        if not child.is_dir() or not child.name.startswith("session-"):
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue

