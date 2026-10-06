"""เชื่อมต่อโมเดล Claude สำหรับวิเคราะห์ภาพโรคพืชและตอบแชท

ใช้ Anthropic Python SDK อย่างเดียว ไม่เรียก HTTP เอง
- วิเคราะห์ภาพ: ใช้ structured output เพื่อให้ได้ JSON ที่ตรวจสอบได้
- แชท: ใช้ streaming เพื่อให้ผู้ใช้เห็นคำตอบทันที
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from .config import SETTINGS

logger = logging.getLogger(__name__)

# header/พารามิเตอร์สำหรับ server-side fallback เมื่อคำขอถูกปฏิเสธโดยตัวกรองความปลอดภัย
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_plant_image": {
            "type": "boolean",
            "description": "ภาพนี้เป็นภาพพืช ส่วนของพืช หรือแปลงปลูกหรือไม่",
        },
        "crop_guess": {
            "type": "string",
            "description": "พืชที่เห็นในภาพ เช่น แตงโม พืชตระกูลแตงอื่น หรือไม่ใช่พืช",
        },
        "plant_part": {
            "type": "string",
            "enum": [
                "ใบ",
                "เถา/ลำต้น",
                "ผล",
                "ราก",
                "ดอก",
                "ทั้งต้น",
                "ทั้งแปลง",
                "ไม่ชัดเจน",
            ],
        },
        "image_quality": {
            "type": "object",
            "properties": {
                "usable": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "advice": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["usable", "issues", "advice"],
            "additionalProperties": False,
        },
        "observations": {
            "type": "array",
            "description": "สิ่งที่มองเห็นจริงในภาพ อธิบายเชิงรูปธรรม เช่น สี รูปร่างแผล ตำแหน่ง",
            "items": {"type": "string"},
        },
        "candidates": {
            "type": "array",
            "description": "โรคหรือสาเหตุที่เป็นไปได้ เรียงจากมากไปน้อย ไม่เกิน 4 รายการ",
            "items": {
                "type": "object",
                "properties": {
                    "disease_id": {
                        "type": "string",
                        "description": "รหัส id จากรายการที่ให้ไว้ ถ้าไม่ตรงกับรายการใดให้ใส่ unknown",
                    },
                    "name_th": {"type": "string"},
                    "confidence": {
                        "type": "integer",
                        "description": "ความมั่นใจเป็นเปอร์เซ็นต์ 0-100",
                    },
                    "evidence": {
                        "type": "array",
                        "description": "หลักฐานในภาพที่สนับสนุนข้อสรุปนี้",
                        "items": {"type": "string"},
                    },
                    "against": {
                        "type": "array",
                        "description": "สิ่งที่ขัดแย้งหรือยังไม่ยืนยันข้อสรุปนี้",
                        "items": {"type": "string"},
                    },
                },
                "required": ["disease_id", "name_th", "confidence", "evidence", "against"],
                "additionalProperties": False,
            },
        },
        "severity": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["น้อย", "ปานกลาง", "รุนแรง", "รุนแรงมาก", "ประเมินไม่ได้"],
                },
                "affected_area_percent": {"type": "integer"},
                "spread_risk": {"type": "string"},
            },
            "required": ["level", "affected_area_percent", "spread_risk"],
            "additionalProperties": False,
        },
        "immediate_actions": {
            "type": "array",
            "description": "สิ่งที่ควรทำทันทีวันนี้ถึงพรุ่งนี้ (ไม่ต้องระบุอัตรายา ระบบจะเติมจากคลังความรู้)",
            "items": {"type": "string"},
        },
        "need_more_checks": {
            "type": "array",
            "description": "สิ่งที่ต้องไปตรวจเพิ่มในแปลงเพื่อยืนยัน เช่น พลิกใต้ใบ ตัดเถาดูท่อน้ำ ขุดราก",
            "items": {"type": "string"},
        },
        "summary_th": {
            "type": "string",
            "description": "สรุปผลวิเคราะห์เป็นภาษาไทยที่เกษตรกรอ่านเข้าใจได้ 3-6 ประโยค",
        },
    },
    "required": [
        "is_plant_image",
        "crop_guess",
        "plant_part",
        "image_quality",
        "observations",
        "candidates",
        "severity",
        "immediate_actions",
        "need_more_checks",
        "summary_th",
    ],
    "additionalProperties": False,
}

DIAGNOSE_SYSTEM = """คุณเป็นนักวิชาการโรคพืช (plant pathologist) ผู้เชี่ยวชาญพืชตระกูลแตงในเขตร้อนชื้นของประเทศไทย
หน้าที่ของคุณคือวิเคราะห์ภาพอาการผิดปกติของแตงโมและระบุสาเหตุที่เป็นไปได้ พร้อมบอกว่าวิเคราะห์จากหลักฐานอะไรในภาพ

หลักการทำงานที่ต้องยึดอย่างเคร่งครัด:
1. อธิบายสิ่งที่เห็นจริงในภาพก่อน แล้วจึงสรุปสาเหตุจากหลักฐานนั้น ห้ามเดาอาการที่ไม่ปรากฏในภาพ
2. เลือก disease_id จากรายการที่ให้ไว้เท่านั้น ถ้าไม่มีรายการใดตรงให้ใช้ "unknown"
3. ให้ค่า confidence ตามหลักฐานที่มีจริง ไม่ปั้นให้สูงเกินจริง เกณฑ์:
   - 85-100 เมื่อเห็นอาการวินิจฉัยเฉพาะ (pathognomonic) ชัดเจน เช่น ขุยราใต้ใบของราน้ำค้าง ยางสีอำพันของโรคยางไหล เม็ดสเคลอโรเทียมที่โคนต้น
   - 60-84 เมื่ออาการเข้ากันดีแต่ยังมีโรคอื่นที่เป็นไปได้
   - 35-59 เมื่ออาการกว้างและแยกจากโรคอื่นได้ไม่ชัด
   - ต่ำกว่า 35 เมื่อภาพไม่ชัด มุมไม่เหมาะ หรือข้อมูลไม่พอ และต้องบอกวิธีถ่ายภาพหรือจุดที่ต้องตรวจเพิ่ม
4. เสนอได้หลายความเป็นไปได้ และต้องระบุทั้งหลักฐานที่สนับสนุน (evidence) และสิ่งที่ยังขัดแย้ง (against)
5. ต้องแยกโรคติดเชื้อออกจากอาการขาดธาตุอาหาร พิษสารเคมี ความเสียหายจากน้ำและแดด และความเสียหายจากแมลงหรือไร
   เพราะเกษตรกรมักพ่นยาฆ่าเชื้อกับอาการที่ไม่ใช่โรคติดเชื้อ ทำให้เสียเงินเปล่า
6. ถ้าโรคที่สงสัยต้องยืนยันด้วยการตรวจที่มองจากภาพนี้ไม่ได้ (เช่น ตัดเถาดูท่อลำเลียง ขุดรากดูปุ่มปม พลิกใต้ใบ
   หรือต้องใช้ชุดตรวจไวรัสในห้องปฏิบัติการ) ต้องระบุไว้ใน need_more_checks ให้ชัด
7. ห้ามระบุชื่อสารเคมีพร้อมอัตราการใช้ใน immediate_actions เพราะระบบจะเติมข้อมูลยาและอัตราจากคลังความรู้ที่ตรวจทานแล้วเอง
   ให้เน้นการจัดการเชิงปฏิบัติ เช่น เก็บส่วนที่เป็นโรคออก ระบายน้ำ หยุดให้น้ำแบบพ่นฝอย แยกต้นที่เป็นโรค
8. ตอบเป็นภาษาไทยที่เกษตรกรอ่านเข้าใจง่าย หลีกเลี่ยงศัพท์วิชาการที่ไม่จำเป็น แต่คงความแม่นยำทางวิชาการ
9. ถ้าภาพไม่ใช่ภาพพืชหรือไม่เกี่ยวกับโรคพืช ให้ตั้ง is_plant_image เป็น false และอธิบายอย่างสุภาพใน summary_th"""

CHAT_SYSTEM = """คุณเป็น "ผู้ช่วยหมอพืชแตงโม" ผู้ช่วยตอบคำถามเกษตรกรไทยเรื่องโรคแตงโม ยาป้องกันกำจัดโรคพืช ปุ๋ย และการดูแลแปลง

หลักการตอบ:
1. ตอบจากข้อมูลในคลังความรู้ที่แนบมาเป็นหลัก ถ้าคลังความรู้ไม่มีข้อมูลให้บอกตรง ๆ ว่าไม่มีข้อมูลในระบบ แล้วแนะนำว่าควรปรึกษาที่ไหน
2. ตอบเป็นภาษาไทยที่กระชับและนำไปใช้ได้จริง เริ่มด้วยคำตอบตรงประเด็น แล้วค่อยให้รายละเอียด
3. เมื่อแนะนำสารเคมี ต้องระบุชื่อสามัญ (ไม่ใช่เฉพาะชื่อการค้า) อัตราต่อน้ำ 20 ลิตร กลุ่มสาร FRAC หรือ IRAC และระยะเก็บเกี่ยวปลอดภัย
   และต้องใช้ตัวเลขจากคลังความรู้ที่แนบมาเท่านั้น ห้ามประมาณอัตราขึ้นเอง ถ้าไม่มีข้อมูลให้บอกว่าต้องดูจากฉลาก
4. เตือนเสมอว่าต้องอ่านฉลาก ใช้อัตราตามฉลาก สวมอุปกรณ์ป้องกัน และเคารพระยะเก็บเกี่ยวปลอดภัย
5. ย้ำเรื่องการสลับกลุ่มสารเพื่อลดการดื้อยา และเรื่องการไม่พ่นสารที่เป็นพิษต่อผึ้งในช่วงดอกบาน
6. ถ้าคำถามคลุมเครือหรืออาการที่เล่ามาเข้าได้หลายโรค ให้ถามกลับ 1-2 คำถามที่ช่วยแยกโรคได้มากที่สุด แทนการเดา
7. ถ้าผู้ใช้ถามเรื่องโรคไวรัส ต้องบอกชัดว่าไม่มียารักษาและต้องจัดการที่แมลงพาหะกับการถอนต้น
8. ถ้าผู้ใช้ถามเรื่องนอกขอบเขตการเกษตร ให้ปฏิเสธอย่างสุภาพและชวนกลับมาที่เรื่องแตงโม
9. งานนี้ผู้ใช้รออ่านคำตอบอยู่ตรงหน้า (latency-sensitive) ให้เริ่มเขียนคำตอบที่มองเห็นได้ทันที"""


class LLMUnavailable(RuntimeError):
    """ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY หรือยังไม่ได้ติดตั้ง SDK"""


def _client():
    if not SETTINGS.api_key:
        raise LLMUnavailable("ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - ขึ้นกับการติดตั้ง
        raise LLMUnavailable("ยังไม่ได้ติดตั้งแพ็กเกจ anthropic (pip install anthropic)") from exc
    return anthropic.Anthropic(api_key=SETTINGS.api_key, max_retries=2, timeout=180.0)


def _fallback_kwargs() -> dict[str, Any]:
    """เปิด server-side fallback เมื่อคำขอถูกปฏิเสธโดยตัวกรองความปลอดภัย"""
    if not SETTINGS.enable_fallback:
        return {}
    return {
        "extra_headers": {"anthropic-beta": _FALLBACK_BETA},
        "extra_body": {"fallbacks": "default"},
    }


def _first_text(message: Any) -> str:
    for block in message.content:
        if getattr(block, "type", "") == "text":
            return block.text
    return ""


def _refusal_message(message: Any) -> str:
    details = getattr(message, "stop_details", None)
    category = getattr(details, "category", None) or "ไม่ระบุ"
    return (
        "ระบบไม่สามารถประมวลผลคำขอนี้ได้ (ตัวกรองความปลอดภัยปฏิเสธ, หมวด: "
        f"{category}) กรุณาลองส่งภาพหรือคำถามที่เกี่ยวกับโรคพืชโดยตรง"
    )


def analyze_image(
    image_b64: str,
    media_type: str,
    disease_catalog: str,
    user_context: str = "",
) -> dict[str, Any]:
    """วิเคราะห์ภาพด้วยโมเดล Claude แล้วคืนผลเป็น dict ตาม DIAGNOSIS_SCHEMA"""
    client = _client()
    context_block = user_context.strip() or "ผู้ใช้ไม่ได้ให้ข้อมูลประกอบเพิ่มเติม"
    prompt = (
        "รายการโรค อาการผิดปกติ และศัตรูพืชที่ระบบรองรับ (ให้เลือก disease_id จากรายการนี้เท่านั้น):\n"
        f"{disease_catalog}\n\n"
        "ข้อมูลประกอบจากผู้ใช้:\n"
        f"{context_block}\n\n"
        "กรุณาวิเคราะห์ภาพนี้ตามหลักการที่กำหนด และตอบเป็น JSON ตามโครงสร้างที่ระบุ"
    )
    request: dict[str, Any] = {
        "model": SETTINGS.model,
        "max_tokens": 16000,
        "system": [
            {
                "type": "text",
                "text": DIAGNOSE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "output_config": {
            "effort": SETTINGS.effort_diagnose,
            "format": {"type": "json_schema", "schema": DIAGNOSIS_SCHEMA},
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    message = _create_with_fallback(client, request)

    if getattr(message, "stop_reason", "") == "refusal":
        raise LLMUnavailable(_refusal_message(message))

    text = _first_text(message)
    if not text:
        raise LLMUnavailable("โมเดลไม่ได้ส่งผลวิเคราะห์กลับมา กรุณาลองใหม่")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable("ผลวิเคราะห์ที่ได้ไม่ใช่ JSON ที่ถูกต้อง กรุณาลองใหม่") from exc

    usage = getattr(message, "usage", None)
    data["_meta"] = {
        "model": getattr(message, "model", SETTINGS.model),
        "effort": SETTINGS.effort_diagnose,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return data


def _create_with_fallback(client: Any, request: dict[str, Any]) -> Any:
    """เรียก messages.create พร้อม server-side fallback และถอย fallback ออกถ้า API ไม่รับ"""
    extra = _fallback_kwargs()
    if extra:
        try:
            return client.messages.create(**request, **extra)
        except Exception as exc:  # noqa: BLE001 - ครอบคลุมทุกข้อผิดพลาดจาก API
            if _is_bad_request(exc):
                logger.warning("server-side fallback ไม่พร้อมใช้งาน จึงเรียกซ้ำโดยไม่ใช้: %s", exc)
            else:
                raise
    return client.messages.create(**request)


def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (400, 404, 422)


def chat_stream(
    user_message: str,
    history: list[dict[str, str]],
    context: str,
) -> Iterator[str]:
    """ตอบคำถามแบบ streaming คืนข้อความทีละชิ้น"""
    client = _client()
    messages: list[dict[str, Any]] = []
    for turn in history[-10:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    context_block = context.strip() or "ไม่พบข้อมูลที่ตรงกับคำถามนี้ในคลังความรู้"
    messages.append(
        {
            "role": "user",
            "content": (
                "ข้อมูลจากคลังความรู้ที่เกี่ยวข้องกับคำถาม (ใช้เป็นแหล่งอ้างอิงหลัก):\n"
                "<คลังความรู้>\n"
                f"{context_block}\n"
                "</คลังความรู้>\n\n"
                f"คำถามจากผู้ใช้: {user_message}"
            ),
        }
    )

    request: dict[str, Any] = {
        "model": SETTINGS.model,
        "max_tokens": 8192,
        "system": [
            {
                "type": "text",
                "text": CHAT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "output_config": {"effort": SETTINGS.effort_chat},
        "messages": messages,
    }
    extra = _fallback_kwargs()

    def _run(kwargs: dict[str, Any]) -> Iterator[str]:
        with client.messages.stream(**request, **kwargs) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()
            if getattr(final, "stop_reason", "") == "refusal":
                yield "\n\n" + _refusal_message(final)

    if extra:
        try:
            yield from _run(extra)
            return
        except Exception as exc:  # noqa: BLE001
            if not _is_bad_request(exc):
                raise
            logger.warning("server-side fallback ไม่พร้อมใช้งานในโหมดแชท: %s", exc)
    yield from _run({})
