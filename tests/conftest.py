"""ค่าตั้งต้นร่วมสำหรับชุดทดสอบ"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.knowledge import load_knowledge  # noqa: E402


@pytest.fixture(scope="session")
def kb():
    return load_knowledge()


def _jpeg(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="session")
def leaf_with_spots() -> bytes:
    """ภาพจำลอง: ใบเขียวที่มีจุดแผลสีน้ำตาลกระจาย"""
    arr = np.zeros((480, 480, 3), dtype=np.uint8)
    arr[:, :] = (70, 150, 70)
    for cx, cy in [(80, 80), (160, 120), (260, 200), (340, 90), (120, 320), (220, 400), (400, 380)]:
        arr[cy - 14 : cy + 14, cx - 14 : cx + 14] = (80, 45, 25)
    return _jpeg(arr)


@pytest.fixture(scope="session")
def healthy_leaf() -> bytes:
    """ภาพจำลอง: ใบเขียวสมบูรณ์"""
    rng = np.random.default_rng(7)
    arr = np.zeros((420, 420, 3), dtype=np.uint8)
    arr[:, :] = (62, 148, 64)
    noise = rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    return _jpeg(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def white_coated_leaf() -> bytes:
    """ภาพจำลอง: ใบเขียวที่มีคราบขาวคล้ายราแป้ง"""
    arr = np.zeros((420, 420, 3), dtype=np.uint8)
    arr[:, :] = (70, 150, 70)
    arr[60:260, 60:300] = (242, 242, 240)
    return _jpeg(arr)


@pytest.fixture(scope="session")
def leaf_on_soil_background() -> bytes:
    """ภาพจำลองที่ใกล้เคียงภาพถ่ายจากแปลงจริง: ใบมีคราบขาวบนพื้นหลังดินและเงา

    ใช้ยืนยันว่าดินและเงาในพื้นหลังไม่ถูกนับเป็นจุดแผล ซึ่งเคยทำให้
    ภาพราแป้งจริงถูกวินิจฉัยเป็นโรคใบจุดผิด
    """
    rng = np.random.default_rng(11)
    arr = np.zeros((420, 420, 3), dtype=np.uint8)
    # พื้นหลังดินสีน้ำตาลที่มีเงาเข้มกระจาย
    arr[:, :] = (120, 95, 70)
    for _ in range(90):
        y, x = rng.integers(0, 400, size=2)
        size = int(rng.integers(6, 16))
        arr[y : y + size, x : x + size] = (38, 28, 20)
    # ใบเขียวตรงกลางภาพ
    arr[90:330, 90:330] = (62, 142, 62)
    # คราบขาวแบบราแป้งบนใบ
    for cy, cx, r in [(150, 150, 26), (210, 240, 32), (270, 160, 22), (180, 290, 18)]:
        arr[cy - r : cy + r, cx - r : cx + r] = (222, 228, 214)
    return _jpeg(arr)
