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


def _local_mean(mask: np.ndarray, radius: int) -> np.ndarray:
    """ค่าเฉลี่ยของมาสก์ในหน้าต่างสี่เหลี่ยม คำนวณด้วย integral image จึงเร็วพอ

    ใช้ปิดรูเล็ก ๆ ในใบ (เช่น จุดแผล) ให้ถูกนับรวมอยู่ในบริเวณใบ
    และตัดจุดสีเขียวเล็ก ๆ ที่กระจายอยู่ในพื้นหลังออก
    """
    m = mask.astype(np.float32)
    integral = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    height, width = m.shape
    ys, xs = np.arange(height), np.arange(width)
    y0, y1 = np.clip(ys - radius, 0, height), np.clip(ys + radius + 1, 0, height)
    x0, x1 = np.clip(xs - radius, 0, width), np.clip(xs + radius + 1, 0, width)
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return total / np.maximum(area, 1.0)


def _plant_region(plant_mask: np.ndarray) -> np.ndarray:
    """คืนบริเวณที่น่าจะเป็นเนื้อเยื่อพืชในภาพ

    ภาพถ่ายจากแปลงจริงมีดิน เงา และวัชพืชปนอยู่มาก ถ้านับจุดสีเข้มทั้งภาพว่าเป็นแผล
    จะได้ผลผิดอย่างมาก (เงาระหว่างก้อนดินถูกนับเป็นจุดแผลหลายสิบจุด)
    จึงต้องจำกัดการวิเคราะห์ไว้เฉพาะบริเวณใบก่อน
    """
    radius = max(3, int(min(plant_mask.shape) * 0.045))
    filled = _local_mean(plant_mask, radius) > 0.45
    region = filled | plant_mask
    # ถ้ามีก้อนใหญ่ก้อนเดียวที่ชัดเจน ให้ใช้เฉพาะก้อนนั้นเป็นวัตถุหลักของภาพ
    total = float(region.size)
    components = _label_blobs(region)
    if components:
        largest = max(components, key=lambda b: b["pixels"])
        if largest["pixels"] >= total * 0.18:
            return _largest_component_mask(region)
    return region


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """คืนมาสก์เฉพาะก้อนที่ใหญ่ที่สุดของมาสก์ที่ให้มา"""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    best_pixels: list[tuple[int, int]] = []
    for y0 in range(height):
        for x0 in range(width):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            queue: deque[tuple[int, int]] = deque([(y0, x0)])
            visited[y0, x0] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels
    out = np.zeros_like(mask, dtype=bool)
    if best_pixels:
        ys, xs = zip(*best_pixels)
        out[np.array(ys), np.array(xs)] = True
    return out


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

    # จำกัดการวิเคราะห์ไว้เฉพาะบริเวณที่เป็นเนื้อเยื่อพืช มิฉะนั้นดินและเงาในภาพถ่ายจริง
    # จะถูกนับเป็นจุดแผลจำนวนมากจนวินิจฉัยผิด
    plant_seed = green | yellow
    plant = _plant_region(plant_seed) if plant_seed.sum() > total * 0.02 else plant_seed
    plant_px = max(float(plant.sum()), 1.0)
    plant_coverage = float(plant.sum()) / total * 100

    whitish = whitish & plant

    # ตรวจ "คราบซีด" แบบปรับตามเนื้อใบรอบข้าง แทนการใช้ค่าคงที่
    # เพราะราแป้งคือบริเวณที่ซีดกว่าและสว่างกว่าเนื้อใบปกติ ซึ่งค่าคงที่จับได้ไม่ดีในทุกสภาพแสง
    if plant.sum() > 100:
        sat_median = float(np.median(sat[plant]))
        val_median = float(np.median(val[plant]))
        pale = plant & (sat < sat_median * 0.62) & (val > val_median * 1.02)
    else:
        pale = np.zeros_like(plant)
    pale_blobs = _label_blobs(pale)
    pale_patch_px = sum(b["pixels"] for b in pale_blobs if b["pixels"] >= plant_px * 0.0015)

    lesion_mask = (brown | dark) & ~whitish & ~pale & plant
    blobs = _label_blobs(lesion_mask)
    blob_pixels = sum(b["pixels"] for b in blobs)
    big_blobs = [b for b in blobs if b["pixels"] >= plant_px * 0.004]

    # ความสม่ำเสมอของสีในบริเวณที่เป็นพืช ใช้แยก "เหลืองทั้งใบ" จาก "จุดแผลกระจาย"
    hue_std = float(np.std(hue[plant_seed & plant])) if (plant_seed & plant).sum() > 50 else 0.0

    # สัดส่วนความเสียหายที่ขอบใบเทียบกับกลางใบ ใช้ประเมินอาการไหม้จากขอบใบ
    # วัดจากขอบของบริเวณใบจริง ไม่ใช่ขอบของกรอบภาพ
    edge_radius = max(2, int(min(plant.shape) * 0.05))
    interior = _local_mean(plant, edge_radius) > 0.92
    leaf_edge = plant & ~interior
    edge_px = max(float(leaf_edge.sum()), 1.0)
    interior_px = max(float((plant & interior).sum()), 1.0)
    border_damage = float((lesion_mask & leaf_edge).sum()) / edge_px
    center_damage = float((lesion_mask & interior).sum()) / interior_px

    return {
        # เปอร์เซ็นต์ทั้งหมดคิดเทียบกับ "พื้นที่พืชในภาพ" ไม่ใช่ทั้งภาพ
        "pct_green": round(float((green & plant).sum()) / plant_px * 100, 2),
        "pct_yellow": round(float((yellow & plant).sum()) / plant_px * 100, 2),
        "pct_brown": round(float((brown & plant).sum()) / plant_px * 100, 2),
        "pct_dark": round(float((dark & plant).sum()) / plant_px * 100, 2),
        "pct_whitish": round(float(whitish.sum()) / plant_px * 100, 2),
        "pct_pale": round(float(pale.sum()) / plant_px * 100, 2),
        "pct_pale_patch": round(pale_patch_px / plant_px * 100, 2),
        "pct_red": round(float((red & plant).sum()) / plant_px * 100, 2),
        "pct_soil": round(float(soil.sum()) / total * 100, 2),
        "pct_lesion": round(blob_pixels / plant_px * 100, 2),
        "plant_coverage": round(plant_coverage, 2),
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
    """ให้คะแนนกลุ่มอาการจากคุณลักษณะของภาพ คืน (disease_id, score, evidence)

    น้ำหนักปรับจากการวัดภาพโรคพืชจริงจากแปลง (ชุด PlantDoc) โดยพบว่าสิ่งที่แยกกลุ่มอาการ
    ได้ดีที่สุดไม่ใช่ค่าสัมบูรณ์ แต่เป็น "สัดส่วนระหว่างหลักฐาน" เช่น
      - คราบซีดสว่างเทียบกับใบเหลือง ต่างกันราว 16 เท่าระหว่างราแป้งกับโรคใบจุด
      - คราบซีดสว่างเทียบกับพื้นที่แผล ต่างกันราว 9 เท่า
    การใช้สัดส่วนยังทนต่อความต่างของแสงและระยะถ่ายภาพได้ดีกว่าการใช้เกณฑ์ตายตัว
    """
    out: list[tuple[str, float, list[str]]] = []

    def add(disease_id: str, score: float, evidence: list[str]) -> None:
        if score > 1:
            out.append((disease_id, round(score, 2), evidence))

    green = f["pct_green"]
    yellow = f["pct_yellow"]
    brown = f["pct_brown"]
    whitish = f["pct_whitish"]
    pale_patch = f["pct_pale_patch"]
    lesion = f["pct_lesion"]
    blobs = f["lesion_blobs"]
    hue_std = f["hue_std_plant"]
    plant_coverage = f["plant_coverage"]
    edge = f["border_damage_ratio"]
    center = f["center_damage_ratio"]

    if plant_coverage < 4:
        return out

    # ตัวหน่วงสำหรับอาการที่ควรมี "แผลเป็นจุดน้อย" เช่น ราน้ำค้างและการขาดธาตุอาหาร
    # ภาพโรคใบจุดมีจุดแผลจำนวนมาก (ค่ากลางราว 49 จุด) จึงถูกหน่วงลงอย่างชัดเจน
    few_spot_damp = 1.0 / (1.0 + blobs / 22.0)

    # ---------- คราบซีดสว่างบนใบ เทียบกับใบเหลืองและแผล -> ราแป้ง ----------
    if whitish >= 0.5 and green > 15:
        powdery = 60.0 * whitish / (whitish + yellow * 0.9 + lesion * 0.5 + 1.0)
        add(
            "powdery_mildew",
            powdery,
            [
                f"พบคราบซีดสว่างบนเนื้อใบ {whitish:.1f}% ของพื้นที่ใบ "
                f"ขณะที่ใบเหลืองมีเพียง {yellow:.1f}% และแผลแห้ง {lesion:.1f}%",
                "สัดส่วนแบบนี้เข้ากับราแป้งที่เกาะอยู่บนผิวใบ มากกว่าโรคที่ทำลายเนื้อใบ",
            ],
        )

    # ---------- จุดแผลกระจายบนใบ -> กลุ่มโรคใบจุด ----------
    if lesion >= 1.2 and blobs >= 4:
        spot = 60.0 * (lesion * 0.7 + brown * 0.6) / (
            lesion * 0.7 + brown * 0.6 + whitish * 2.5 + 1.5
        )
        add(
            "anthracnose",
            spot,
            [
                f"พบแผลสีน้ำตาลเข้มถึงดำแยกกัน {blobs} จุด รวม {lesion:.1f}% ของพื้นที่ใบ",
                "ลักษณะเป็นจุดแผลกระจายบนเนื้อใบ ซึ่งเข้ากับกลุ่มโรคใบจุดจากเชื้อรา",
            ],
        )
        add(
            "alternaria_blight",
            spot * 0.85,
            ["แผลกระจายเป็นจุด ต้องดูว่ามีวงซ้อนเป็นชั้นในแผลหรือไม่เพื่อแยกอัลเทอร์นาเรีย"],
        )
        if blobs >= 12 and lesion < 12:
            add(
                "cercospora_leaf_spot",
                spot * 0.62,
                ["แผลมีจำนวนมากแต่แต่ละจุดเล็ก ซึ่งพบได้ในโรคใบจุดเซอร์โคสปอรา"],
            )

    # ---------- ปื้นเหลืองโดยมีจุดแผลไม่มาก -> ราน้ำค้าง ----------
    if yellow >= 4 and green > 15:
        downy = 60.0 * yellow / (yellow + whitish * 2.0 + 2.0) * few_spot_damp
        if hue_std > 22:
            downy *= 1.25
        add(
            "downy_mildew",
            downy,
            [
                f"พบปื้นสีเหลือง {yellow:.1f}% ของพื้นที่ใบปนกับเนื้อใบเขียว โดยมีจุดแผลแยกกัน {blobs} จุด",
                "ต้องพลิกดูใต้ใบเพื่อหาขุยราสีเทาอมม่วง จึงจะแยกจากการขาดธาตุอาหารได้แน่นอน",
            ],
        )

    # ---------- เหลืองสม่ำเสมอโดยแผลน้อยมาก -> ขาดธาตุอาหาร ----------
    if yellow >= 6 and lesion < 3.5:
        deficiency = 60.0 * yellow / (yellow + lesion * 3.0 + 3.0) * few_spot_damp
        add(
            "nitrogen_deficiency",
            deficiency,
            [
                f"ใบเหลืองเป็นบริเวณกว้าง {yellow:.1f}% โดยมีแผลแห้งน้อยเพียง {lesion:.1f}%",
                "ไม่พบแผลที่มีขอบเขตชัด จึงเข้ากับอาการขาดธาตุอาหารมากกว่าโรคติดเชื้อ",
            ],
        )
        add(
            "magnesium_deficiency",
            deficiency * 0.7,
            ["ต้องดูว่าเส้นใบยังเขียวอยู่หรือไม่ เพื่อแยกการขาดแมกนีเซียมจากการขาดไนโตรเจน"],
        )

    # ---------- ไหม้จากขอบใบเข้ามา -> ขาดโพแทสเซียมหรือดินเค็ม ----------
    if edge > 10 and edge > center * 2.5 and lesion >= 1:
        edge_score = min(edge * 1.1, 42)
        add(
            "potassium_deficiency",
            edge_score,
            [
                f"ความเสียหายกระจุกที่ขอบใบ ({edge:.1f}%) มากกว่าเนื้อใบด้านใน ({center:.1f}%) "
                "ซึ่งเข้ากับอาการไหม้จากขอบใบ"
            ],
        )
        add(
            "salinity_stress",
            edge_score * 0.6,
            ["ขอบใบไหม้อาจเกิดจากปุ๋ยเข้มข้นเกินหรือดินเค็ม ให้ตรวจว่าเพิ่งใส่ปุ๋ยหรือไม่"],
        )

    # ---------- บริเวณซีดขาวบนผล -> ผลไหม้แดด ----------
    if whitish > 8 and f["pct_red"] > 2:
        add(
            "sunscald",
            min(whitish * 1.5, 42),
            ["พบบริเวณผิวซีดขาวบนผล ซึ่งเข้ากับอาการผลไหม้แดด"],
        )

    # ---------- แผลผืนใหญ่ต่อเนื่อง -> กลุ่มเน่า ----------
    if f["large_lesion_blobs"] >= 1 and lesion > 14 and blobs <= 8:
        add(
            "phytophthora_blight",
            min(lesion * 1.4, 45),
            [
                f"พบบริเวณเน่าเสียหายเป็นผืนใหญ่ต่อเนื่อง {lesion:.1f}% ของพื้นที่ใบ",
                "ต้องดูว่าเนื้อเยื่อเน่าเละฉ่ำน้ำหรือแห้ง และมีราขาวฟูหรือไม่",
            ],
        )

    # ---------- ภาพที่ดูปกติ ----------
    # ต้องไม่มีคราบซีดสว่างเหลืออยู่เลย มิฉะนั้นจะไปทับกับกฎราแป้งซึ่งใช้เกณฑ์เดียวกัน
    if green > 70 and lesion < 1.5 and yellow < 5 and whitish < 0.6:
        add(
            "healthy",
            45.0,
            [f"พื้นที่ใบเขียวสมบูรณ์ {green:.1f}% และพบความเสียหายน้อยกว่า 1.5%"],
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
    if features["plant_coverage"] < 8:
        issues.append("ไม่พบพื้นที่ที่เป็นเนื้อเยื่อพืชชัดเจนในภาพ")
        advice.append("ถ่ายให้เห็นใบ ผล หรือเถาเต็มกรอบภาพมากขึ้น")

    plant_part = "ไม่ชัดเจน"
    if features["pct_red"] > 6 and features["pct_green"] < 25:
        plant_part = "ผล"
    elif features["plant_coverage"] > 60 and features["pct_soil"] < 25:
        plant_part = "ใบ"
    elif features["plant_coverage"] > 10:
        plant_part = "ทั้งแปลง"

    top_name = candidates[0]["disease_id"] if candidates else "ไม่สามารถสรุปได้"
    return {
        "is_plant_image": features["plant_coverage"] > 4,
        "crop_guess": "ไม่สามารถระบุชนิดพืชได้ในโหมดออฟไลน์",
        "plant_part": plant_part,
        "image_quality": {"usable": usable, "issues": issues, "advice": advice},
        "observations": [
            f"บริเวณที่เป็นเนื้อเยื่อพืชกินพื้นที่ {features['plant_coverage']}% ของภาพ "
            "(ส่วนที่เหลือเป็นดินหรือพื้นหลัง ซึ่งไม่ถูกนำมาวิเคราะห์)",
            f"ในพื้นที่ใบ: สีเขียว {features['pct_green']}% สีเหลือง {features['pct_yellow']}% "
            f"สีน้ำตาล {features['pct_brown']}% คราบซีดสว่าง {features['pct_whitish']}%",
            f"พบแผลแยกกัน {features['lesion_blobs']} จุด คิดเป็น {features['pct_lesion']}% ของพื้นที่ใบ",
            f"ความเสียหายที่ขอบใบ {features['border_damage_ratio']}% "
            f"และเนื้อใบด้านใน {features['center_damage_ratio']}%",
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
