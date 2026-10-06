"""FastAPI application: ระบบ AI วินิจฉัยโรคแตงโมและผู้ช่วยตอบคำถาม"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import local_model
from .chat import answer_stream, offline_answer
from .config import SETTINGS
from .diagnose import ImageError, diagnose
from .knowledge import load_knowledge
from .schemas import ChatRequest, ChatResponse, DiagnoseResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("melon")

KB = load_knowledge()


@asynccontextmanager
async def lifespan(_: FastAPI):
    stats = KB.stats()
    logger.info(
        "โหลดคลังความรู้แล้ว: โรค %s รายการ, ผลิตภัณฑ์ %s รายการ, FAQ %s รายการ",
        stats["diseases"],
        stats["products"],
        stats["faq"],
    )
    logger.info(
        "โหมดการทำงาน: %s | โมเดล: %s | โมเดลที่เทรนเอง: %s",
        "เชื่อมต่อ Claude" if SETTINGS.ai_enabled else "ออฟไลน์ (ไม่มี ANTHROPIC_API_KEY)",
        SETTINGS.model,
        "พร้อมใช้" if local_model.available() else "ไม่มี",
    )
    yield


app = FastAPI(
    title="ระบบ AI วินิจฉัยโรคแตงโม",
    description=(
        "วิเคราะห์โรคแตงโมจากภาพด้วย Claude vision พร้อมคลังความรู้โรคพืช "
        "ยาป้องกันกำจัดโรคพืช และโปรแกรมปุ๋ยภาษาไทย"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- หน้าเว็บ


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    page = SETTINGS.web_dir / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="ไม่พบหน้าเว็บ")
    return FileResponse(page)


# ---------------------------------------------------------------- API


@app.get("/api/health")
def health() -> dict[str, Any]:
    """สถานะระบบและข้อมูลคลังความรู้"""
    return {
        "status": "ok",
        "ai_enabled": SETTINGS.ai_enabled,
        "model": SETTINGS.model if SETTINGS.ai_enabled else None,
        "local_model": local_model.available(),
        "mode": "claude" if SETTINGS.ai_enabled else "offline",
        "knowledge": KB.stats(),
        "limits": {
            "max_upload_mb": SETTINGS.max_upload_mb,
            "image_max_edge": SETTINGS.image_max_edge,
        },
    }


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def api_diagnose(
    image: UploadFile = File(..., description="ภาพอาการของพืช"),
    context: str = Form("", description="ข้อมูลประกอบ เช่น อายุต้น สภาพอากาศ สารที่พ่นล่าสุด"),
) -> DiagnoseResponse:
    """วิเคราะห์โรคจากภาพ แล้วคืนผลพร้อมแผนการจัดการจากคลังความรู้"""
    raw = await image.read()
    try:
        result = diagnose(
            raw_image=raw,
            kb=KB,
            content_type=image.content_type,
            user_context=context[:2000],
        )
    except ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("วิเคราะห์ภาพไม่สำเร็จ")
        raise HTTPException(status_code=500, detail=f"วิเคราะห์ภาพไม่สำเร็จ: {exc}") from exc
    return DiagnoseResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest) -> ChatResponse:
    """ถามตอบแบบไม่สตรีม (เหมาะกับการเรียกจากสคริปต์หรือทดสอบ)"""
    history = [turn.model_dump() for turn in payload.history]
    if not SETTINGS.ai_enabled:
        answer, sources = offline_answer(payload.message, KB)
        return ChatResponse(
            answer=answer, sources=sources, engine="offline_retrieval", ai_enabled=False
        )
    stream, sources, engine = answer_stream(
        payload.message, history, KB, payload.diagnosis_context
    )
    answer = "".join(stream)
    return ChatResponse(answer=answer, sources=sources, engine=engine, ai_enabled=True)


@app.post("/api/chat/stream")
def api_chat_stream(payload: ChatRequest) -> StreamingResponse:
    """ถามตอบแบบสตรีมผ่าน Server-Sent Events"""
    history = [turn.model_dump() for turn in payload.history]
    stream, sources, engine = answer_stream(
        payload.message, history, KB, payload.diagnosis_context
    )

    def event_source():
        yield _sse({"type": "meta", "engine": engine, "sources": sources})
        try:
            for chunk in stream:
                if chunk:
                    yield _sse({"type": "delta", "text": chunk})
        except Exception as exc:  # noqa: BLE001
            logger.exception("สตรีมคำตอบไม่สำเร็จ")
            yield _sse({"type": "error", "message": f"เกิดข้อผิดพลาด: {exc}"})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/diseases")
def api_diseases(
    group: str | None = Query(None, description="กรองตามกลุ่ม เช่น fungal, viral, abiotic, pest"),
    part: str | None = Query(None, description="กรองตามส่วนของพืช เช่น ใบ ผล ราก"),
) -> dict[str, Any]:
    """รายการโรคและอาการผิดปกติทั้งหมดในคลังความรู้"""
    items = []
    for rec in KB.diseases.values():
        if group and rec["group_id"] != group:
            continue
        if part and not any(part in p for p in rec.get("affected_parts", [])):
            continue
        items.append(
            {
                "id": rec["id"],
                "name_th": rec.get("name_th", ""),
                "name_en": rec.get("name_en", ""),
                "aliases_th": rec.get("aliases_th", []),
                "group_id": rec["group_id"],
                "group_th": rec["group_th"],
                "pathogen": rec.get("pathogen", ""),
                "pathogen_type": rec.get("pathogen_type", ""),
                "severity": rec.get("severity", ""),
                "severity_score": rec.get("severity_score"),
                "affected_parts": rec.get("affected_parts", []),
                "image_cues": rec.get("image_cues", []),
                "is_infectious": rec.get("pathogen_type") not in {"abiotic", "insect", "mite"},
            }
        )
    items.sort(key=lambda item: (item["group_id"], -(item["severity_score"] or 0)))
    return {"count": len(items), "groups": KB.groups, "items": items}


@app.get("/api/diseases/{disease_id}")
def api_disease_detail(disease_id: str) -> dict[str, Any]:
    """รายละเอียดโรคหนึ่งรายการพร้อมแผนการจัดการที่ประกอบจากคลังความรู้"""
    plan = KB.treatment_plan(disease_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="ไม่พบโรคนี้ในคลังความรู้")
    rec = KB.disease(disease_id) or {}
    return {
        **plan,
        "aliases_th": rec.get("aliases_th", []),
        "image_cues": rec.get("image_cues", []),
        "not_image_cues": rec.get("not_image_cues", []),
        "stages_at_risk": rec.get("stages_at_risk", []),
        "survival": rec.get("survival", ""),
    }


@app.get("/api/pesticides")
def api_pesticides(
    type: str | None = Query(None, description="ชนิด เช่น fungicide, insecticide, biological"),
    target: str | None = Query(None, description="รหัสโรคหรือศัตรูพืชที่ต้องการควบคุม"),
    organic: bool | None = Query(None, description="เฉพาะที่ใช้ได้ในระบบอินทรีย์"),
) -> dict[str, Any]:
    """รายการยาและสารป้องกันกำจัดศัตรูพืชในคลังความรู้"""
    items = []
    for rec in KB.pesticide_data["items"]:
        if type and rec.get("type") != type:
            continue
        if target and target not in (rec.get("targets") or []):
            continue
        if organic is not None and bool(rec.get("organic_ok")) != organic:
            continue
        items.append(rec)
    return {
        "count": len(items),
        "disclaimer": KB.pesticide_data.get("disclaimer", ""),
        "safety_rules": KB.pesticide_data.get("safety_rules", []),
        "banned_or_restricted_th": KB.pesticide_data.get("banned_or_restricted_th", {}),
        "items": items,
    }


@app.get("/api/pesticides/{product_id}")
def api_pesticide_detail(product_id: str) -> dict[str, Any]:
    rec = KB.product(product_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="ไม่พบผลิตภัณฑ์นี้ในคลังความรู้")
    targets = [
        {"id": t, "name_th": (KB.disease(t) or {}).get("name_th", t)}
        for t in rec.get("targets", []) or []
    ]
    return {**rec, "target_details": targets}


@app.get("/api/fertilizers")
def api_fertilizers() -> dict[str, Any]:
    """ฐานข้อมูลปุ๋ยและโปรแกรมปุ๋ยตามระยะการเจริญเติบโต"""
    return KB.fertilizer_data


@app.get("/api/crop")
def api_crop() -> dict[str, Any]:
    """ข้อมูลการปลูกและดูแลแตงโม"""
    return KB.crop


@app.get("/api/faq")
def api_faq() -> dict[str, Any]:
    return {"count": len(KB.faq), "items": KB.faq}


@app.get("/api/sources")
def api_sources() -> dict[str, Any]:
    return {"count": len(KB.sources), "items": KB.sources}


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1, description="คำค้น"),
    limit: int = Query(8, ge=1, le=30),
    kind: str | None = Query(None, description="จำกัดประเภท: disease, product, faq, crop"),
) -> dict[str, Any]:
    """ค้นหาข้ามคลังความรู้ทั้งหมด"""
    kinds = [kind] if kind else None
    hits = KB.index.search(q, limit=limit, kinds=kinds)
    return {
        "query": q,
        "count": len(hits),
        "items": [
            {
                "kind": doc.kind,
                "id": doc.doc_id,
                "title": doc.title,
                "score": round(score, 2),
                "excerpt": doc.text[:280],
            }
            for doc, score in hits
        ],
    }


if SETTINGS.web_dir.is_dir():
    app.mount("/static", StaticFiles(directory=SETTINGS.web_dir), name="static")
