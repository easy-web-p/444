"""ประสานงานการวินิจฉัยโรคจากภาพ

ลำดับการทำงาน:
1. ตรวจและย่อขนาดภาพ
2. วิเคราะห์ด้วยโมเดล Claude (ถ้ามี API key) หรือโหมดออฟไลน์
3. ผสมผลจากโมเดล ONNX ที่เทรนเอง ถ้ามี
4. ตรวจสอบรหัสโรคกับคลังความรู้ และประกอบแผนการรักษาจากคลังความรู้
   (อัตรายาและปุ๋ยมาจากไฟล์ JSON ที่ตรวจทานแล้ว ไม่ใช่จากโมเดล)
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image, UnidentifiedImageError

from . import local_model
from .config import SETTINGS
from .knowledge import Knowledge
from .llm import LLMUnavailable, analyze_image, api_error_message
from .vision_offline import analyze_offline

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

DISCLAIMER = (
    "ผลวิเคราะห์นี้เป็นการคัดกรองเบื้องต้นด้วย AI ไม่ใช่การวินิจฉัยที่ยืนยันแล้ว "
    "อัตรายาและปุ๋ยที่แสดงเป็นแนวทางจากคลังความรู้ของระบบ ต้องอ่านฉลากผลิตภัณฑ์และใช้อัตราตามฉลาก "
    "ตรวจสอบสถานะการขึ้นทะเบียนของสารกับกรมวิชาการเกษตร และเคารพระยะเก็บเกี่ยวปลอดภัย (PHI) ทุกครั้ง "
    "หากความมั่นใจต่ำหรือความเสียหายสูง ควรปรึกษานักวิชาการเกษตรในพื้นที่หรือส่งตัวอย่างตรวจที่คลินิกพืช"
)


class ImageError(ValueError):
    """ภาพที่ส่งมาใช้งานไม่ได้"""


def prepare_image(raw: bytes, content_type: str | None = None) -> tuple[str, str, dict[str, Any]]:
    """ตรวจสอบและย่อภาพ คืน (base64, media_type, ข้อมูลภาพ)"""
    limit = SETTINGS.max_upload_mb * 1024 * 1024
    if not raw:
        raise ImageError("ไม่พบไฟล์ภาพ")
    if len(raw) > limit:
        raise ImageError(f"ไฟล์ใหญ่เกิน {SETTINGS.max_upload_mb} MB กรุณาย่อภาพก่อนอัปโหลด")
    if content_type and content_type.split(";")[0].strip() not in ALLOWED_TYPES:
        raise ImageError("รองรับเฉพาะไฟล์ภาพ JPEG, PNG, WebP หรือ HEIC")

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            original = img.size
            img.thumbnail((SETTINGS.image_max_edge, SETTINGS.image_max_edge), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88, optimize=True)
            resized = img.size
    except (UnidentifiedImageError, OSError) as exc:
        if content_type and "hei" in content_type.lower():
            raise ImageError(
                "อ่านไฟล์ HEIC ไม่สำเร็จ เครื่องที่รันระบบอาจยังไม่รองรับรูปแบบนี้ "
                "ให้ติดตั้งส่วนเสริมด้วย pip install pillow-heif หรือบันทึกภาพเป็น JPEG ก่อนอัปโหลด"
            ) from exc
        raise ImageError("อ่านไฟล์ภาพไม่สำเร็จ ไฟล์อาจเสียหายหรือไม่ใช่ภาพ") from exc

    data = buf.getvalue()
    info = {
        "original_size": f"{original[0]}x{original[1]}",
        "sent_size": f"{resized[0]}x{resized[1]}",
        "bytes_original": len(raw),
        "bytes_sent": len(data),
    }
    return base64.standard_b64encode(data).decode("ascii"), "image/jpeg", info


def _normalize_candidates(
    raw_candidates: list[dict[str, Any]], kb: Knowledge
) -> list[dict[str, Any]]:
    """ตรวจรหัสโรคกับคลังความรู้ ปรับค่าความมั่นใจให้อยู่ในช่วง 0-100 และเรียงลำดับ"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_candidates or []:
        disease_id = (item.get("disease_id") or "").strip()
        rec = kb.disease(disease_id)
        if rec is None and disease_id not in {"", "unknown", "healthy"}:
            rec = kb.resolve_disease(item.get("name_th") or disease_id)
        key = rec["id"] if rec else (disease_id or item.get("name_th") or "unknown")
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = int(round(float(item.get("confidence") or 0)))
        except (TypeError, ValueError):
            confidence = 0
        out.append(
            {
                "disease_id": rec["id"] if rec else None,
                "name_th": (rec or {}).get("name_th") or item.get("name_th") or "ไม่ระบุ",
                "name_en": (rec or {}).get("name_en", ""),
                "group_th": (rec or {}).get("group_th", ""),
                "confidence": max(0, min(confidence, 100)),
                "evidence": [str(x) for x in (item.get("evidence") or [])][:6],
                "against": [str(x) for x in (item.get("against") or [])][:6],
                "in_knowledge_base": rec is not None,
                "is_infectious": (
                    rec.get("pathogen_type") not in {"abiotic", "insect", "mite"} if rec else None
                ),
            }
        )
    out.sort(key=lambda c: c["confidence"], reverse=True)
    return out[:4]


def _merge_local_model(
    candidates: list[dict[str, Any]], predictions: list[dict[str, Any]], kb: Knowledge
) -> list[dict[str, Any]]:
    """ผสมผลจากโมเดลที่เทรนเองเข้ากับผลหลัก (เพิ่มความมั่นใจเมื่อสองแหล่งเห็นตรงกัน)"""
    if not predictions:
        return candidates
    by_id = {c["disease_id"]: c for c in candidates if c["disease_id"]}
    for pred in predictions:
        disease_id = pred["disease_id"]
        rec = kb.disease(disease_id)
        note = f"โมเดลภาพที่เทรนเองให้ความมั่นใจ {pred['confidence']}%"
        if disease_id in by_id:
            cand = by_id[disease_id]
            cand["confidence"] = min(
                100, int(round(cand["confidence"] * 0.75 + pred["confidence"] * 0.35))
            )
            cand["evidence"].append(note + " ซึ่งตรงกับผลวิเคราะห์หลัก")
        elif rec and pred["confidence"] >= 50:
            candidates.append(
                {
                    "disease_id": rec["id"],
                    "name_th": rec.get("name_th", ""),
                    "name_en": rec.get("name_en", ""),
                    "group_th": rec.get("group_th", ""),
                    "confidence": int(round(pred["confidence"] * 0.6)),
                    "evidence": [note + " แต่ผลวิเคราะห์หลักไม่ได้เสนอโรคนี้"],
                    "against": ["ยังไม่ได้รับการยืนยันจากผลวิเคราะห์หลัก"],
                    "in_knowledge_base": True,
                    "is_infectious": rec.get("pathogen_type")
                    not in {"abiotic", "insect", "mite"},
                }
            )
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates[:4]


def _safety_block(kb: Knowledge, plan: dict[str, Any] | None) -> dict[str, Any]:
    pest = kb.pesticide_data
    phi_values = [
        c["phi_days"]
        for c in (plan or {}).get("chemical", [])
        if isinstance(c.get("phi_days"), int)
    ]
    bee_risky = [
        c["name_th"]
        for c in (plan or {}).get("chemical", [])
        if "สูง" in str(c.get("bee_toxicity", ""))
    ]
    return {
        "rules": pest.get("safety_rules", [])[:6],
        "max_phi_days": max(phi_values) if phi_values else None,
        "bee_risky_products": bee_risky,
        "banned_note": (pest.get("banned_or_restricted_th") or {}).get("note", ""),
    }


def _confidence_guidance(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["ไม่สามารถสรุปสาเหตุจากภาพนี้ได้ กรุณาถ่ายภาพใหม่ตามคำแนะนำ"]
    top = candidates[0]["confidence"]
    notes: list[str] = []
    if top < 35:
        notes.append(
            "ความมั่นใจต่ำกว่า 35% ยังไม่ควรตัดสินใจซื้อสารเคมีจากผลนี้ "
            "ให้ถ่ายภาพเพิ่มและตรวจอาการตามรายการที่แนะนำก่อน"
        )
    elif top < 60:
        notes.append(
            "ความมั่นใจอยู่ในระดับปานกลาง ควรยืนยันด้วยการตรวจอาการเพิ่มเติมก่อนใช้สารที่มีราคาสูง"
        )
    if len(candidates) > 1 and candidates[0]["confidence"] - candidates[1]["confidence"] <= 15:
        notes.append(
            f"มีสองสาเหตุที่ความมั่นใจใกล้เคียงกัน ({candidates[0]['name_th']} และ "
            f"{candidates[1]['name_th']}) ให้ใช้จุดแยกโรคในส่วน 'โรคที่อาการคล้ายกัน' เพื่อตัดสิน"
        )
    non_infectious = [c for c in candidates[:2] if c.get("is_infectious") is False]
    if non_infectious:
        notes.append(
            f"สาเหตุที่สงสัย ({non_infectious[0]['name_th']}) ไม่ใช่โรคติดเชื้อ "
            "การพ่นสารกำจัดเชื้อราหรือแบคทีเรียจะไม่ช่วย ต้องแก้ที่การจัดการน้ำ ปุ๋ย หรือแมลงตามที่ระบุ"
        )
    return notes


def diagnose(
    raw_image: bytes,
    kb: Knowledge,
    content_type: str | None = None,
    user_context: str = "",
) -> dict[str, Any]:
    """วินิจฉัยโรคจากภาพและคืนผลพร้อมแผนการจัดการ"""
    image_b64, media_type, image_info = prepare_image(raw_image, content_type)
    raw_resized = base64.standard_b64decode(image_b64)

    engines_used: list[str] = []
    engine = "offline_heuristic"
    result: dict[str, Any]
    llm_error: str | None = None

    if SETTINGS.ai_enabled:
        try:
            result = analyze_image(
                image_b64=image_b64,
                media_type=media_type,
                disease_catalog=kb.disease_catalog_for_prompt(),
                user_context=user_context,
            )
            engine = "claude_vision"
            engines_used.append("claude_vision")
        except LLMUnavailable as exc:
            llm_error = str(exc)
            logger.warning("เรียกโมเดล AI ไม่สำเร็จ จึงใช้โหมดออฟไลน์: %s", exc)
            result = analyze_offline(raw_resized)
            engines_used.append("offline_heuristic")
        except Exception as exc:  # noqa: BLE001
            hint = api_error_message(exc)
            if hint:
                llm_error = hint
                logger.warning("เรียกโมเดล AI ไม่สำเร็จ จึงใช้โหมดออฟไลน์: %s", hint)
            else:
                llm_error = f"เกิดข้อผิดพลาดในการเรียกโมเดล AI: {exc}"
                logger.exception("เรียกโมเดล AI ไม่สำเร็จ")
            result = analyze_offline(raw_resized)
            engines_used.append("offline_heuristic")
    else:
        result = analyze_offline(raw_resized)
        engines_used.append("offline_heuristic")

    candidates = _normalize_candidates(result.get("candidates", []), kb)

    if local_model.available():
        predictions = local_model.predict(raw_resized)
        if predictions:
            engines_used.append("local_onnx")
            candidates = _merge_local_model(candidates, predictions, kb)

    top = candidates[0] if candidates else None
    plan = kb.treatment_plan(top["disease_id"]) if top and top["disease_id"] else None
    alt_plan = None
    if len(candidates) > 1 and candidates[1]["disease_id"]:
        gap = candidates[0]["confidence"] - candidates[1]["confidence"]
        if gap <= 20:
            alt_plan = kb.treatment_plan(candidates[1]["disease_id"])

    guidance = _confidence_guidance(candidates)
    severity = result.get("severity") or {}
    quality = result.get("image_quality") or {}

    meta = {
        **(result.get("_meta") or {}),
        "image": image_info,
        "ai_enabled": SETTINGS.ai_enabled,
        "local_model": local_model.available(),
    }
    if llm_error:
        meta["llm_error"] = llm_error

    return {
        "ok": True,
        "engine": engine,
        "engines_used": engines_used,
        "is_plant_image": bool(result.get("is_plant_image", True)),
        "crop_guess": str(result.get("crop_guess") or ""),
        "plant_part": str(result.get("plant_part") or ""),
        "image_quality": quality,
        "observations": [str(x) for x in (result.get("observations") or [])][:8],
        "candidates": candidates,
        "severity": severity,
        "immediate_actions": [str(x) for x in (result.get("immediate_actions") or [])][:8],
        "need_more_checks": [str(x) for x in (result.get("need_more_checks") or [])][:8],
        "summary_th": str(result.get("summary_th") or ""),
        "treatment": plan,
        "alternative_treatment": alt_plan,
        "safety": {**_safety_block(kb, plan), "guidance": guidance},
        "disclaimer": DISCLAIMER,
        "meta": meta,
    }
