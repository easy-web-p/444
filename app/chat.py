"""ตอบคำถามแบบแชท ทั้งโหมดที่ใช้โมเดล AI และโหมดออฟไลน์ (ค้นคืนจากคลังความรู้)"""

from __future__ import annotations

from typing import Any, Iterator

from .config import SETTINGS
from .knowledge import Knowledge
from .llm import LLMUnavailable, chat_stream

OFFLINE_HEADER = (
    "ขณะนี้ระบบทำงานในโหมดออฟไลน์ (ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY) "
    "จึงตอบด้วยการค้นคืนข้อมูลจากคลังความรู้โดยตรง ไม่ได้เรียบเรียงใหม่ด้วย AI\n"
)

NO_RESULT = (
    "ไม่พบข้อมูลที่ตรงกับคำถามนี้ในคลังความรู้ของระบบ\n\n"
    "ลองถามใหม่โดยใช้คำที่เจาะจงขึ้น เช่น ระบุอาการที่เห็น (ใบเหลือง จุดน้ำตาล ยางไหล ผลเน่า) "
    "ส่วนของพืชที่ผิดปกติ (ใบ เถา ผล ราก) และระยะการเจริญเติบโตของต้น "
    "หรือเลือกดูรายการโรคและรายการยาจากเมนูคลังความรู้"
)


def _format_disease(rec: dict[str, Any], kb: Knowledge) -> str:
    plan = kb.treatment_plan(rec["id"]) or {}
    lines = [f"## {rec.get('name_th','')} ({rec.get('name_en','')})", f"กลุ่ม: {rec.get('group_th','')}"]
    if rec.get("pathogen"):
        lines.append(f"สาเหตุ: {rec['pathogen']}")
    symptoms = rec.get("symptoms") or {}
    for part, items in symptoms.items():
        if items:
            lines.append(f"\nอาการที่{part}:")
            lines.extend(f"- {item}" for item in items[:4])
    if plan.get("urgent"):
        lines.append("\nสิ่งที่ควรทำทันที:")
        lines.extend(f"- {item}" for item in plan["urgent"][:5])
    if plan.get("chemical"):
        lines.append("\nสารเคมีที่ใช้ได้ (อ่านฉลากและใช้อัตราตามฉลากเสมอ):")
        for item in plan["chemical"][:5]:
            phi = f" | ระยะเก็บเกี่ยวปลอดภัย {item['phi_days']} วัน" if item.get("phi_days") is not None else ""
            lines.append(
                f"- {item.get('name_th','')} ({item.get('group','')}) อัตรา {item.get('rate','')}"
                f" ทุก {item.get('interval','-')} โดย{item.get('mode','')}{phi}"
            )
    if plan.get("biological"):
        lines.append("\nชีวภัณฑ์และทางเลือกที่ปลอดภัยกว่า:")
        for item in plan["biological"][:4]:
            lines.append(f"- {item.get('name_th','')} อัตรา {item.get('rate','')} — {item.get('note','')}")
    if plan.get("cultural"):
        lines.append("\nการจัดการแปลง:")
        lines.extend(f"- {item}" for item in plan["cultural"][:5])
    if plan.get("fertilizer_advice"):
        lines.append("\nคำแนะนำปุ๋ย:")
        lines.extend(f"- {item}" for item in plan["fertilizer_advice"][:4])
    if plan.get("lookalikes"):
        lines.append("\nโรคที่อาการคล้ายกันและจุดแยก:")
        for look in plan["lookalikes"][:3]:
            lines.append(f"- {look.get('name_th','')}: {look.get('key_difference','')}")
    if rec.get("notes"):
        lines.append(f"\nข้อสังเกตสำคัญ: {rec['notes']}")
    return "\n".join(lines)


def _format_product(rec: dict[str, Any]) -> str:
    lines = [f"## {rec.get('name_th','')} ({rec.get('name_en','')})"]
    if rec.get("group"):
        lines.append(f"กลุ่มสาร: {rec['group']}")
    if rec.get("nutrients"):
        lines.append(f"ธาตุอาหาร: {rec['nutrients']}")
    if rec.get("mode"):
        lines.append(f"ลักษณะการออกฤทธิ์: {rec['mode']}")
    if rec.get("rate_20l"):
        lines.append(f"อัตราต่อน้ำ 20 ลิตร: {rec['rate_20l']}")
    if rec.get("rate_rai"):
        lines.append(f"อัตราต่อไร่: {rec['rate_rai']}")
    if rec.get("phi_days") is not None:
        lines.append(f"ระยะเก็บเกี่ยวปลอดภัย: {rec['phi_days']} วัน")
    if rec.get("bee_toxicity"):
        lines.append(f"ความเป็นพิษต่อผึ้ง: {rec['bee_toxicity']}")
    if rec.get("resistance_risk"):
        lines.append(f"ความเสี่ยงการดื้อยา: {rec['resistance_risk']}")
    cautions = rec.get("mix_cautions") or rec.get("cautions") or []
    if cautions:
        lines.append("ข้อควรระวัง:")
        lines.extend(f"- {item}" for item in cautions)
    if rec.get("notes"):
        lines.append(f"หมายเหตุ: {rec['notes']}")
    return "\n".join(lines)


def offline_answer(message: str, kb: Knowledge) -> tuple[str, list[dict[str, Any]]]:
    """ตอบคำถามโดยไม่ใช้โมเดล AI ด้วยการค้นคืนและจัดรูปข้อมูลจากคลังความรู้"""
    hits = kb.index.search(message, limit=5)
    if not hits:
        return NO_RESULT, []
    # ตัดผลลัพธ์ที่คะแนนต่ำกว่า 35% ของอันดับหนึ่งออก เพื่อลดข้อมูลที่ไม่เกี่ยวข้อง
    cutoff = hits[0][1] * 0.35
    hits = [hit for hit in hits if hit[1] >= cutoff][:3]

    parts: list[str] = [OFFLINE_HEADER]
    sources: list[dict[str, Any]] = []
    for doc, score in hits:
        sources.append({"kind": doc.kind, "id": doc.doc_id, "title": doc.title, "score": round(score, 2)})
        if doc.kind == "disease":
            rec = kb.disease(doc.doc_id)
            if rec:
                parts.append(_format_disease(rec, kb))
                continue
        if doc.kind == "product":
            rec = kb.product(doc.doc_id)
            if rec:
                parts.append(_format_product(rec))
                continue
        if doc.kind == "faq":
            item = next((f for f in kb.faq if f["id"] == doc.doc_id), None)
            if item:
                parts.append(f"## {item['q']}\n{item['a']}")
                continue
        parts.append(f"## {doc.title}\n{doc.text[:900]}")

    parts.append(
        "\n---\nคำเตือน: อ่านฉลากผลิตภัณฑ์และใช้อัตราตามฉลากเสมอ สวมอุปกรณ์ป้องกัน "
        "สลับกลุ่มสารเพื่อลดการดื้อยา และเคารพระยะเก็บเกี่ยวปลอดภัย"
    )
    return "\n\n".join(parts), sources


def answer_stream(
    message: str, history: list[dict[str, str]], kb: Knowledge
) -> tuple[Iterator[str], list[dict[str, Any]], str]:
    """คืน (ตัววนข้อความ, แหล่งอ้างอิง, ชื่อ engine)"""
    context, sources = kb.context_for_chat(message, limit=6)
    if not SETTINGS.ai_enabled:
        text, offline_sources = offline_answer(message, kb)
        return iter([text]), offline_sources, "offline_retrieval"

    def _gen() -> Iterator[str]:
        try:
            yield from chat_stream(message, history, context)
        except LLMUnavailable as exc:
            text, _ = offline_answer(message, kb)
            yield f"[ไม่สามารถเรียกโมเดล AI ได้: {exc}]\n\n{text}"
        except Exception as exc:  # noqa: BLE001
            text, _ = offline_answer(message, kb)
            yield f"[เกิดข้อผิดพลาดในการเรียกโมเดล AI: {exc}]\n\n{text}"

    return _gen(), sources, "claude_chat"
