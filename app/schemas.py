"""โครงสร้างข้อมูลสำหรับ API"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    engine: str
    ai_enabled: bool


class CandidateOut(BaseModel):
    disease_id: str | None = None
    name_th: str
    name_en: str = ""
    group_th: str = ""
    confidence: int
    evidence: list[str] = Field(default_factory=list)
    against: list[str] = Field(default_factory=list)
    in_knowledge_base: bool = False
    is_infectious: bool | None = None


class DiagnoseResponse(BaseModel):
    ok: bool
    engine: str
    engines_used: list[str] = Field(default_factory=list)
    is_plant_image: bool
    crop_guess: str = ""
    plant_part: str = ""
    image_quality: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    candidates: list[CandidateOut] = Field(default_factory=list)
    severity: dict[str, Any] = Field(default_factory=dict)
    immediate_actions: list[str] = Field(default_factory=list)
    need_more_checks: list[str] = Field(default_factory=list)
    summary_th: str = ""
    treatment: dict[str, Any] | None = None
    alternative_treatment: dict[str, Any] | None = None
    safety: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
