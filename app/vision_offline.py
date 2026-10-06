"""ตัววิเคราะห์ภาพแบบออฟไลน์ (ไม่ต้องใช้ API key)

ใช้การวิเคราะห์สีและลักษณะแผลด้วยวิธี computer vision แบบดั้งเดิม
(HSV segmentation + connected components) แล้วเทียบกับโปรไฟล์อาการของกลุ่มโรค

ข้อจำกัดที่ต้องสื่อสารกับผู้ใช้อย่างตรงไปตรงมา: วิธีนี้แยกได้เพียง "กลุ่มอาการ"
เช่น ใบเหลืองทั้งใบ จุดแผลกระจาย หรือคราบขาว ไม่สามารถระบุชนิดเชื้อได้แม่นยำ
ความมั่นใจจึงถูกจำกัดไม่ให้เกิน 45% และต้องให้ผู้ใช้ยืนยันด้วยวิธีอื่นเสมอ
"""

from __future__ import annotations

import io
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

MAX_EDGE = 384
MIN_BLOB_PIXELS = 12
CONFIDENCE_CAP = 45


def _rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """แปลง RGB (0-1) เป็น HSV โดย H หน่วยองศา 0-360, S และ V อยู่ในช่วง 0-1"""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    rmax = mask & (maxc == r)
    gmax = mask & (maxc == g) & ~rmax
    bmax = mask & (maxc == b) & ~rmax & ~gmax
    with np.errstate(invalid="ignore", divide="ignore"):
        hue[rmax] = (60 * ((g[rmax] - b[rmax]) / delta[rmax])) % 360
        hue[gmax] = 60 * ((b[gmax] - r[gmax]) / delta[gmax]) + 120
        hue[bmax] = 60 * ((r[bmax] - g[bmax]) / delta[bmax]) + 240
    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    return hue, sat, maxc


def _label_blobs(mask: np.ndarray) -> list[dict[str, float]]:
    """หา connected components แบบ 4-connectivity และคืนสถิติของแต่ละก้อน"""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    blobs: list[dict[str, float]] = []
    for y0 in range(height):
        row = mask[y0]
        for x0 in range(width):
            if not row[x0] or visited[y0, x0]:
                continue
            queue: deque[tuple[int, int]] = deque([(y0, x0)])
            visited[y0, x0] = True
            pixels = 0
            min_y = max_y = y0
            min_x = max_x = x0
            while queue:
                y, x = queue.popleft()
                pixels += 1
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if pixels < MIN_BLOB_PIXELS:
                continue
            box_h = max_y - min_y + 1
            box_w = max_x - min_x + 1
            fill = pixels / float(box_h * box_w)
            aspect = box_w / float(box_h) if box_h else 1.0
            blobs.append(
                {
                    "pixels": float(pixels),
                    "fill_ratio": float(fill),
                    "aspect": float(aspect),
                    "touches_border": float(
                        min_y == 0 or min_x == 0 or max_y == height - 1 or max_x == width - 1
                    ),
                }
            )
    return blobs


def extract_features(image_bytes: bytes) -> dict[str, Any]:
    """คำนวณคุณลักษณะของภาพที่ใช้ในการจำแนกกลุ่มอาการ"""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_EDGE, MAX_EDGE))
        arr = np.asarray(img, dtype=np.float32) / 255.0

    hue, sat, val = _rgb_to_hsv(arr)
    total = float(hue.size)

    green = (hue >= 65) & (hue <= 170) & (sat > 0.18) & (val > 0.12)
    yellow = (hue >= 35) & (hue < 65) & (sat > 0.25) & (val > 0.35)
    brown = (((hue >= 10) & (hue < 45)) | (hue >= 345)) & (sat > 0.15) & (val > 0.08) & (val < 0.62)
    dark = val <= 0.18
    whitish = (sat < 0.16) & (val > 0.68)
    red = (((hue >= 330) | (hue <= 12)) & (sat > 0.35) & (val > 0.3))
    soil = ((hue >= 15) & (hue <= 40)) & (sat >= 0.12) & (sat <= 0.45) & (val >= 0.3) & (val <= 0.75)

    lesion_mask = (brown | dark) & ~whitish
    blobs = _label_blobs(lesion_mask)
    blob_pixels = sum(b["pixels"] for b in blobs)
    big_blobs = [b for b in blobs if b["pixels"] >= total * 0.002]

    # ความสม่ำเสมอของสีในบริเวณที่เป็นพืช ใช้แยก "เหลืองทั้งใบ" จาก "จุดแผลกระจาย"
    plant_mask = green | yellow
    plant_px = float(plant_mask.sum())
    hue_std = float(np.std(hue[plant_mask])) if plant_px > 50 else 0.0

    # สัดส่วนความเสียหายที่ขอบภาพ ใช้ประเมินอาการไหม้จากขอบใบ
    border = np.zeros_like(lesion_mask)
    pad = max(2, int(min(lesion_mask.shape) * 0.08))
    border[:pad, :] = border[-pad:, :] = True
    border[:, :pad] = border[:, -pad:] = True
    border_damage = float((lesion_mask & border).sum()) / max(float(border.sum()), 1.0)
    center_damage = float((lesion_mask & ~border).sum()) / max(float((~border).sum()), 1.0)

    return {
        "pct_green": round(float(green.sum()) / total * 100, 2),
        "pct_yellow": round(float(yellow.sum()) / total * 100, 2),
        "pct_brown": round(float(brown.sum()) / total * 100, 2),
        "pct_dark": round(float(dark.sum()) / total * 100, 2),
        "pct_whitish": round(float(whitish.sum()) / total * 100, 2),
        "pct_red": round(float(red.sum()) / total * 100, 2),
        "pct_soil": round(float(soil.sum()) / total * 100, 2),
        "pct_lesion": round(blob_pixels / total * 100, 2),
        "lesion_blobs": len(blobs),
        "large_lesion_blobs": len(big_blobs),
        "mean_blob_fill": round(
            float(np.mean([b["fill_ratio"] for b in blobs])) if blobs else 0.0, 3
        ),
        "hue_std_plant": round(hue_std, 2),
        "border_damage_ratio": round(border_damage * 100, 2),
        "center_damage_ratio": round(center_damage * 100, 2),
        "brightness_mean": round(float(val.mean()), 3),
        "saturation_mean": round(float(sat.mean()), 3),
        "pixels_analyzed": int(total),
    }


def _rule_scores(f: dict[str, Any]) -> list[tuple[str, float, list[str]]]:
    """ให้คะแนนกลุ่มอาการจากคุณลักษณะของภาพ คืน (disease_id, score, evidence)"""
    out: list[tuple[str, float, list[str]]] = []

    def add(disease_id: str, score: float, evidence: list[str]) -> None:
        if score > 0:
            out.append((disease_id, score, evidence))

    green = f["pct_green"]
    yellow = f["pct_yellow"]
    brown = f["pct_brown"]
    whitish = f["pct_whitish"]
    lesion = f["pct_lesion"]
    blobs = f["lesion_blobs"]
    hue_std = f["hue_std_plant"]

    # คราบขาวบนพื้นใบเขียว -> ราแป้ง
    if whitish > 6 and green > 12:
        add(
            "powdery_mildew",
            min(whitish * 2.2, 60) + (10 if whitish > 14 else 0),
            [f"พบพื้นที่สีขาวซีด {whitish:.1f}% ของภาพบนพื้นใบสีเขียว ซึ่งเข้ากับคราบผงราแป้ง"],
        )

    # ใบเหลืองเป็นปื้นร่วมกับแผลแห้ง -> ราน้ำค้าง (ต้องยืนยันด้วยการพลิกใต้ใบ)
    if yellow > 8 and green > 8:
        score = yellow * 1.6 + (12 if brown > 3 else 0) + (10 if hue_std > 25 else 0)
        add(
            "downy_mildew",
            min(score, 65),
            [
                f"พบปื้นสีเหลือง {yellow:.1f}% ของภาพปนกับเนื้อใบเขียว",
                "สีไม่สม่ำเสมอเป็นหย่อม ซึ่งเข้ากับลักษณะปื้นเหลืองของราน้ำค้าง"
                if hue_std > 25
                else "ต้องพลิกดูใต้ใบเพื่อหาขุยราสีเทาอมม่วงจึงจะยืนยันได้",
            ],
        )

    # เหลืองสม่ำเสมอทั้งใบโดยแผลน้อย -> ขาดธาตุอาหาร
    if yellow > 12 and lesion < 6 and hue_std < 28:
        add(
            "nitrogen_deficiency",
            min(yellow * 1.9, 60),
            [
                f"ใบเหลืองเป็นบริเวณกว้าง {yellow:.1f}% โดยมีแผลแห้งน้อย ({lesion:.1f}%)",
                "สีเหลืองค่อนข้างสม่ำเสมอ ซึ่งเข้ากับอาการขาดธาตุอาหารมากกว่าโรคติดเชื้อ",
            ],
        )
        add(
            "magnesium_deficiency",
            min(yellow * 1.4, 45),
            ["ใบเหลืองเป็นบริเวณกว้าง ต้องดูว่าเส้นใบยังเขียวอยู่หรือไม่เพื่อแยกจากการขาดไนโตรเจน"],
        )

    # จุดแผลแยกกันหลายจุด -> กลุ่มโรคใบจุดจากเชื้อรา
    if blobs >= 4 and lesion > 1.5:
        score = min(blobs * 3.0, 40) + min(lesion * 2.0, 25)
        add(
            "anthracnose",
            score,
            [
                f"พบแผลสีน้ำตาลเข้มถึงดำแยกกัน {blobs} จุด รวม {lesion:.1f}% ของภาพ",
                "ลักษณะเป็นจุดแผลกระจาย ซึ่งเข้ากับกลุ่มโรคใบจุดจากเชื้อรา",
            ],
        )
        add(
            "alternaria_blight",
            score * 0.8,
            ["แผลกระจายเป็นจุด ต้องดูว่ามีวงซ้อนเป็นชั้นในแผลหรือไม่เพื่อแยกอัลเทอร์นาเรีย"],
        )

    # ไหม้จากขอบเข้ามา -> ขาดโพแทสเซียมหรือดินเค็ม
    if f["border_damage_ratio"] > 12 and f["border_damage_ratio"] > f["center_damage_ratio"] * 1.6:
        add(
            "potassium_deficiency",
            min(f["border_damage_ratio"] * 1.6, 50),
            [
                f"ความเสียหายกระจุกที่ขอบภาพ ({f['border_damage_ratio']:.1f}%) มากกว่าบริเวณกลาง "
                f"({f['center_damage_ratio']:.1f}%) ซึ่งเข้ากับอาการไหม้จากขอบใบ"
            ],
        )
        add(
            "salinity_stress",
            min(f["border_damage_ratio"] * 1.1, 38),
            ["ขอบใบไหม้อาจเกิดจากปุ๋ยเข้มข้นเกินหรือดินเค็ม ให้ตรวจว่าเพิ่งใส่ปุ๋ยหรือไม่"],
        )

    # บริเวณซีดขาวบนผล -> ผลไหม้แดด
    if whitish > 10 and f["pct_red"] > 2:
        add(
            "sunscald",
            min(whitish * 1.6, 42),
            ["พบบริเวณผิวซีดขาวบนผล ซึ่งเข้ากับอาการผลไหม้แดด"],
        )

    # แผลใหญ่ต่อเนื่องบริเวณกลางภาพ -> กลุ่มเน่า
    if f["large_lesion_blobs"] >= 1 and lesion > 12 and blobs <= 6:
        add(
            "phytophthora_blight",
            min(lesion * 1.5, 45),
            [
                f"พบบริเวณเน่าเสียหายเป็นผืนใหญ่ต่อเนื่อง {lesion:.1f}% ของภาพ",
                "ต้องดูว่าเนื้อเยื่อเน่าเละฉ่ำน้ำหรือแห้ง และมีราขาวฟูหรือไม่",
            ],
        )

    # ภาพที่ดูปกติ
    if green > 45 and lesion < 1.5 and yellow < 6 and whitish < 5:
        add(
            "healthy",
            55.0,
            [f"พื้นที่ใบสีเขียวสมบูรณ์ {green:.1f}% และพบความเสียหายน้อยกว่า 1.5%"],
        )

    return out


def analyze_offline(image_bytes: bytes) -> dict[str, Any]:
    """วิเคราะห์ภาพแบบออฟไลน์ คืนโครงสร้างเดียวกับผลจากโมเดล AI"""
    features = extract_features(image_bytes)
    scored = _rule_scores(features)
    scored.sort(key=lambda item: item[1], reverse=True)

    total = sum(score for _, score, _ in scored) or 1.0
    candidates: list[dict[str, Any]] = []
    for disease_id, score, evidence in scored[:4]:
        confidence = int(round(min(score / total * 100, 100) * CONFIDENCE_CAP / 100))
        candidates.append(
            {
                "disease_id": disease_id,
                "name_th": "",
                "confidence": max(confidence, 5),
                "evidence": evidence,
                "against": ["วิเคราะห์จากสีและลักษณะแผลเท่านั้น ไม่ได้ยืนยันชนิดเชื้อ"],
            }
        )

    usable = features["pixels_analyzed"] > 10000 and features["brightness_mean"] > 0.08
    issues: list[str] = []
    advice: list[str] = []
    if features["brightness_mean"] < 0.2:
        issues.append("ภาพมืดเกินไป")
        advice.append("ถ่ายใหม่ในที่มีแสงธรรมชาติเพียงพอ")
    if features["brightness_mean"] > 0.92:
        issues.append("ภาพสว่างจ้าจนรายละเอียดหาย")
        advice.append("เลี่ยงแสงแดดจัดและไม่ใช้แฟลช")
    if features["pct_green"] + features["pct_yellow"] < 8:
        issues.append("ไม่พบพื้นที่ที่เป็นเนื้อเยื่อพืชชัดเจนในภาพ")
        advice.append("ถ่ายให้เห็นใบ ผล หรือเถาเต็มกรอบภาพมากขึ้น")

    plant_part = "ไม่ชัดเจน"
    if features["pct_red"] > 6 and features["pct_green"] < 25:
        plant_part = "ผล"
    elif features["pct_green"] + features["pct_yellow"] > 25:
        plant_part = "ใบ"
    elif features["pct_soil"] > 30:
        plant_part = "ทั้งแปลง"

    top_name = candidates[0]["disease_id"] if candidates else "ไม่สามารถสรุปได้"
    return {
        "is_plant_image": features["pct_green"] + features["pct_yellow"] > 5,
        "crop_guess": "ไม่สามารถระบุชนิดพืชได้ในโหมดออฟไลน์",
        "plant_part": plant_part,
        "image_quality": {"usable": usable, "issues": issues, "advice": advice},
        "observations": [
            f"พื้นที่สีเขียว {features['pct_green']}% สีเหลือง {features['pct_yellow']}% "
            f"สีน้ำตาล {features['pct_brown']}% สีขาวซีด {features['pct_whitish']}%",
            f"พบแผลแยกกัน {features['lesion_blobs']} จุด คิดเป็น {features['pct_lesion']}% ของภาพ",
            f"ความเสียหายที่ขอบภาพ {features['border_damage_ratio']}% "
            f"และบริเวณกลางภาพ {features['center_damage_ratio']}%",
        ],
        "candidates": candidates,
        "severity": {
            "level": _severity_level(features["pct_lesion"] + features["pct_yellow"] * 0.5),
            "affected_area_percent": int(
                round(min(features["pct_lesion"] + features["pct_yellow"] * 0.5, 100))
            ),
            "spread_risk": "ประเมินไม่ได้ในโหมดออฟไลน์ ต้องดูการกระจายของอาการในแปลงจริง",
        },
        "immediate_actions": [
            "เก็บใบ ผล หรือเถาที่เป็นโรคออกจากแปลงไปทำลายนอกแปลง",
            "หยุดให้น้ำแบบพ่นฝอยเหนือทรงพุ่มและให้น้ำที่โคนต้นแทน",
            "ถ่ายภาพเพิ่มทั้งด้านบนใบ ใต้ใบ และระยะไกลให้เห็นทั้งต้น เพื่อยืนยันการวินิจฉัย",
        ],
        "need_more_checks": [
            "พลิกดูใต้ใบว่ามีขุยรา ตัวไร หรือแมลงหวี่ขาวหรือไม่",
            "ตรวจโคนต้นและข้อเถาว่ามียางไหล รอยเน่า หรือเส้นใยราหรือไม่",
            "ขุดรากต้นที่ผิดปกติดูว่ามีปุ่มปมหรือรากเน่าหรือไม่",
        ],
        "summary_th": (
            "นี่คือผลวิเคราะห์ในโหมดออฟไลน์ ซึ่งใช้การวิเคราะห์สีและลักษณะแผลในภาพเท่านั้น "
            f"กลุ่มอาการที่ใกล้เคียงที่สุดคือ {top_name} "
            "ความแม่นยำของโหมดนี้ต่ำกว่าการวิเคราะห์ด้วยโมเดล AI อย่างมาก "
            "และไม่สามารถระบุชนิดเชื้อได้ กรุณาตั้งค่า ANTHROPIC_API_KEY เพื่อใช้การวิเคราะห์เต็มรูปแบบ "
            "หรือยืนยันผลด้วยการตรวจอาการตามรายการที่แนะนำ"
        ),
        "_meta": {"engine": "offline_heuristic", "features": features},
    }


def _severity_level(damage_percent: float) -> str:
    if damage_percent < 2:
        return "น้อย"
    if damage_percent < 10:
        return "ปานกลาง"
    if damage_percent < 30:
        return "รุนแรง"
    return "รุนแรงมาก"
