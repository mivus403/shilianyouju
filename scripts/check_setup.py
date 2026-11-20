from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.model_client import ModelAPIError, chat_completion_stream  # noqa: E402


def main() -> int:
    settings = load_settings(ROOT)
    print(f"FFmpeg: {settings.ffmpeg_path or '未找到'}")
    print(f"模型: {settings.model}")
    print(f"服务主机: {settings.base_url_host or '未配置'}")
    print(f"API Key: {'已配置（不显示）' if settings.api_key else '未配置'}")
    if not settings.ffmpeg_path:
        print("失败：请配置 FFMPEG_PATH。")
        return 2
    if not settings.model_ready or not settings.api_key or not settings.base_url:
        print("尚不能测试接口，缺少：" + "、".join(settings.missing_model_config))
        return 2
    try:
        response = chat_completion_stream(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            content=[{"type": "text", "text": "这是连接测试。只回复 READY。"}],
            timeout_seconds=settings.request_timeout_seconds,
        )
    except ModelAPIError as exc:
        print(f"接口测试失败：{exc}")
        return 1
    print(f"接口测试成功：{response.text}")
    print(f"实际返回型号：{response.response_model or '接口未提供'}")
    print(f"耗时：{response.elapsed_seconds:.2f} 秒")
    print(f"usage：{response.usage or '接口未提供'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
