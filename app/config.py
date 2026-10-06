"""การตั้งค่าระบบ อ่านจากไฟล์ .env และตัวแปรสภาพแวดล้อม"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"


def load_dotenv(path: Path | None = None) -> None:
    """อ่านไฟล์ .env แบบง่าย โดยไม่ทับค่าที่ตั้งไว้ใน environment อยู่แล้ว"""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class Settings:
    """ค่าตั้งต้นทั้งหมดของแอป"""

    api_key: str = ""
    model: str = "claude-opus-5-5"
    effort_diagnose: str = "high"
    effort_chat: str = "medium"
    enable_fallback: bool = True
    max_upload_mb: int = 12
    image_max_edge: int = 1400
    onnx_model: str = ""
    onnx_labels: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = field(default=DATA_DIR)
    web_dir: Path = field(default=WEB_DIR)

    @property
    def ai_enabled(self) -> bool:
        """มี API key ให้เรียกโมเดลวิเคราะห์ภาพและแชทได้หรือไม่"""
        return bool(self.api_key)

    @property
    def local_model_enabled(self) -> bool:
        return bool(self.onnx_model) and Path(self.onnx_model).is_file()


def get_settings() -> Settings:
    load_dotenv()
    effort_d = (os.environ.get("MELON_EFFORT_DIAGNOSE") or "high").lower()
    effort_c = (os.environ.get("MELON_EFFORT_CHAT") or "medium").lower()
    return Settings(
        api_key=(os.environ.get("ANTHROPIC_API_KEY") or "").strip(),
        model=(os.environ.get("MELON_MODEL") or "claude-opus-5-5").strip(),
        effort_diagnose=effort_d if effort_d in EFFORT_LEVELS else "high",
        effort_chat=effort_c if effort_c in EFFORT_LEVELS else "medium",
        enable_fallback=_bool("MELON_ENABLE_FALLBACK", True),
        max_upload_mb=_int("MELON_MAX_UPLOAD_MB", 12),
        image_max_edge=_int("MELON_IMAGE_MAX_EDGE", 1400),
        onnx_model=(os.environ.get("MELON_ONNX_MODEL") or "").strip(),
        onnx_labels=(os.environ.get("MELON_ONNX_LABELS") or "").strip(),
        host=(os.environ.get("MELON_HOST") or "0.0.0.0").strip(),
        port=_int("MELON_PORT", 8000),
    )


SETTINGS = get_settings()
