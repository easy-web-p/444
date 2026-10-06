"""ช่องต่อโมเดลจำแนกภาพที่เทรนเอง (ONNX) — ใช้เมื่อมีไฟล์โมเดลใน models/

ถ้ามีการตั้งค่า MELON_ONNX_MODEL และติดตั้ง onnxruntime ระบบจะนำผลจากโมเดลนี้
มาผสมกับผลจากโมเดล AI เพื่อเพิ่มความมั่นใจ (ensemble) ดูวิธีเทรนที่ scripts/train_model.py
"""

from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SETTINGS

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _session() -> tuple[Any, list[str], int] | None:
    """โหลด ONNX session และรายการ label (แคชไว้) คืน None ถ้าใช้งานไม่ได้"""
    if not SETTINGS.local_model_enabled:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        logger.info("ไม่พบ onnxruntime จึงข้ามการใช้โมเดลที่เทรนเอง (pip install onnxruntime)")
        return None

    labels_path = Path(SETTINGS.onnx_labels) if SETTINGS.onnx_labels else None
    if labels_path is None or not labels_path.is_file():
        guess = Path(SETTINGS.onnx_model).with_suffix("").as_posix() + ".labels.json"
        labels_path = Path(guess)
    if not labels_path.is_file():
        logger.warning("พบไฟล์โมเดลแต่ไม่พบไฟล์ labels (%s) จึงไม่เปิดใช้งาน", labels_path)
        return None

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = payload["labels"] if isinstance(payload, dict) else list(payload)
    size = int(payload.get("input_size", 224)) if isinstance(payload, dict) else 224
    try:
        sess = ort.InferenceSession(SETTINGS.onnx_model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("โหลดโมเดล ONNX ไม่สำเร็จ: %s", exc)
        return None
    logger.info("เปิดใช้งานโมเดลที่เทรนเอง: %s (%d คลาส)", SETTINGS.onnx_model, len(labels))
    return sess, labels, size


def available() -> bool:
    return _session() is not None


def predict(image_bytes: bytes, top_k: int = 3) -> list[dict[str, Any]]:
    """ทำนายคลาสจากโมเดลที่เทรนเอง คืนรายการ {disease_id, confidence}"""
    loaded = _session()
    if loaded is None:
        return []
    sess, labels, size = loaded

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

    input_name = sess.get_inputs()[0].name
    try:
        logits = np.asarray(sess.run(None, {input_name: tensor})[0]).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("รันโมเดล ONNX ไม่สำเร็จ: %s", exc)
        return []

    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    order = np.argsort(probs)[::-1][: max(top_k, 1)]
    return [
        {
            "disease_id": labels[i] if i < len(labels) else f"class_{i}",
            "confidence": int(round(float(probs[i]) * 100)),
        }
        for i in order
    ]
