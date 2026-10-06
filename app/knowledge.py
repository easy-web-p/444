"""โหลดและให้บริการคลังความรู้โรคแตงโม ยา ปุ๋ย และคำถามที่พบบ่อย

หลักการสำคัญของระบบ: โมเดล AI ทำหน้าที่ "ระบุโรคจากภาพ" เท่านั้น
ส่วน "อัตรายาและปุ๋ย" ดึงจากคลังความรู้ในไฟล์ JSON ที่ตรวจทานแล้ว
เพื่อไม่ให้ตัวเลขอัตราการใช้สารถูกสร้างขึ้นเองโดยโมเดล
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import SETTINGS
from .retrieval import BM25Index, Document, flatten_text

DISEASE_FILES = ("fungal", "oomycete", "bacterial", "viral", "nematode", "abiotic", "pests")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Knowledge:
    """คลังความรู้ทั้งระบบ พร้อมดัชนีค้นคืน"""

    diseases: dict[str, dict[str, Any]]
    groups: dict[str, str]
    products: dict[str, dict[str, Any]]
    fertilizer_data: dict[str, Any]
    pesticide_data: dict[str, Any]
    faq: list[dict[str, Any]]
    crop: dict[str, Any]
    sources: dict[str, Any]
    index: BM25Index

    # ---------- การเข้าถึงข้อมูลพื้นฐาน ----------

    def disease(self, disease_id: str | None) -> dict[str, Any] | None:
        if not disease_id:
            return None
        return self.diseases.get(disease_id)

    def product(self, product_id: str | None) -> dict[str, Any] | None:
        if not product_id:
            return None
        return self.products.get(product_id)

    def resolve_disease(self, text: str) -> dict[str, Any] | None:
        """หาโรคจากชื่อไทย ชื่ออังกฤษ ชื่อพ้อง หรือรหัส"""
        if not text:
            return None
        needle = text.strip().lower()
        for rec in self.diseases.values():
            if needle == rec["id"].lower():
                return rec
            if needle == rec.get("name_th", "").lower():
                return rec
            if needle == (rec.get("name_en") or "").lower():
                return rec
            for alias in rec.get("aliases_th", []) or []:
                if needle == alias.lower():
                    return rec
        hits = self.index.search(text, limit=1, kinds=["disease"])
        if hits and hits[0][1] > 4:
            return self.diseases.get(hits[0][0].payload.get("id", ""))
        return None

    def stats(self) -> dict[str, Any]:
        by_group: dict[str, int] = {}
        for rec in self.diseases.values():
            by_group[rec["group_th"]] = by_group.get(rec["group_th"], 0) + 1
        by_type: dict[str, int] = {}
        for rec in self.products.values():
            by_type[rec.get("type", "อื่น ๆ")] = by_type.get(rec.get("type", "อื่น ๆ"), 0) + 1
        return {
            "diseases": len(self.diseases),
            "diseases_by_group": by_group,
            "products": len(self.products),
            "products_by_type": by_type,
            "faq": len(self.faq),
            "sources": len(self.sources),
            "updated": self.pesticide_data.get("updated", ""),
        }

    # ---------- การประกอบแผนการรักษา ----------

    def _expand_entries(self, entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in entries or []:
            prod = self.product(entry.get("ref"))
            item = {
                "ref": entry.get("ref"),
                "rate": entry.get("rate", ""),
                "interval": entry.get("interval", ""),
                "mode": entry.get("mode", ""),
                "note": entry.get("note", ""),
            }
            if prod:
                item.update(
                    {
                        "name_th": prod.get("name_th", entry.get("ref", "")),
                        "name_en": prod.get("name_en", ""),
                        "group": prod.get("group", prod.get("type", "")),
                        "product_type": prod.get("type", ""),
                        "phi_days": prod.get("phi_days"),
                        "bee_toxicity": prod.get("bee_toxicity", ""),
                        "who_class": prod.get("who_class", ""),
                        "resistance_risk": prod.get("resistance_risk", ""),
                        "mix_cautions": prod.get("cautions") or prod.get("mix_cautions") or [],
                        "organic_ok": prod.get("organic_ok", False),
                        "trade_examples": prod.get("trade_examples", []),
                        "product_note": prod.get("notes", ""),
                    }
                )
            else:
                item["name_th"] = entry.get("ref", "")
            out.append(item)
        return out

    def treatment_plan(self, disease_id: str) -> dict[str, Any] | None:
        """สร้างแผนการจัดการจากคลังความรู้ (อัตรายา/ปุ๋ยมาจากไฟล์ ไม่ใช่จากโมเดล)"""
        rec = self.disease(disease_id)
        if not rec:
            return None
        treat = rec.get("treatment", {})
        return {
            "disease_id": rec["id"],
            "name_th": rec.get("name_th", ""),
            "name_en": rec.get("name_en", ""),
            "group_th": rec.get("group_th", ""),
            "pathogen": rec.get("pathogen", ""),
            "pathogen_type": rec.get("pathogen_type", ""),
            "severity": rec.get("severity", ""),
            "severity_score": rec.get("severity_score"),
            "spread": rec.get("spread", ""),
            "yield_loss": rec.get("yield_loss", ""),
            "affected_parts": rec.get("affected_parts", []),
            "symptoms": rec.get("symptoms", {}),
            "conditions": rec.get("conditions", {}),
            "transmission": rec.get("transmission", []),
            "urgent": treat.get("urgent", []),
            "chemical": self._expand_entries(treat.get("chemical")),
            "biological": self._expand_entries(treat.get("biological")),
            "cultural": treat.get("cultural", []),
            "rotation_note": treat.get("rotation_note", ""),
            "fertilizer_advice": rec.get("fertilizer_advice", []),
            "prevention": rec.get("prevention", []),
            "organic_options": rec.get("organic_options", []),
            "lookalikes": [
                {
                    "id": look.get("id"),
                    "name_th": (self.disease(look.get("id")) or {}).get("name_th")
                    or look.get("name_th", ""),
                    "key_difference": look.get("key_difference", ""),
                }
                for look in rec.get("lookalikes", []) or []
            ],
            "notes": rec.get("notes", ""),
            "refs": [
                {"key": key, **(self.sources.get(key) or {})} for key in rec.get("refs", []) or []
            ],
            "is_infectious": rec.get("pathogen_type")
            not in {"abiotic", "insect", "mite"},
        }

    # ---------- ข้อความสำหรับส่งให้โมเดล ----------

    def disease_catalog_for_prompt(self) -> str:
        """สรุปรายการโรคพร้อมจุดสังเกตจากภาพ สำหรับให้โมเดลเลือกรหัสที่ถูกต้อง

        จัดกลุ่มตาม group_id ที่อยู่ในข้อมูลจริง (ไม่ใช่ชื่อไฟล์) เพื่อไม่ให้มีกลุ่มใดตกหล่น
        """
        lines: list[str] = []
        seen_groups: list[str] = []
        for rec in self.diseases.values():
            if rec["group_id"] not in seen_groups:
                seen_groups.append(rec["group_id"])
        for group_id in seen_groups:
            items = [r for r in self.diseases.values() if r["group_id"] == group_id]
            if not items:
                continue
            lines.append(f"\n## {items[0]['group_th']}")
            for rec in items:
                cues = "; ".join((rec.get("image_cues") or [])[:4])
                parts = ", ".join(rec.get("affected_parts") or [])
                lines.append(
                    f"- id={rec['id']} | {rec.get('name_th','')} ({rec.get('name_en','')}) "
                    f"| ส่วนที่พบ: {parts} | จุดสังเกตจากภาพ: {cues}"
                )
        return "\n".join(lines)

    def context_for_chat(self, query: str, limit: int = 6) -> tuple[str, list[dict[str, Any]]]:
        """ค้นคืนความรู้ที่เกี่ยวข้องกับคำถาม และจัดรูปเป็นบริบทให้โมเดล"""
        hits = self.index.search(query, limit=limit)
        blocks: list[str] = []
        used: list[dict[str, Any]] = []
        for doc, score in hits:
            blocks.append(f"[{doc.kind}:{doc.doc_id}] {doc.title}\n{doc.text.strip()}")
            used.append(
                {
                    "kind": doc.kind,
                    "id": doc.payload.get("id", doc.doc_id),
                    "title": doc.title,
                    "score": round(score, 2),
                }
            )
        return "\n\n---\n\n".join(blocks), used


def _disease_document(rec: dict[str, Any]) -> Document:
    title = f"{rec.get('name_th','')} ({rec.get('name_en','')}) [{rec.get('group_th','')}]"
    searchable = {
        "aliases": rec.get("aliases_th"),
        "pathogen": rec.get("pathogen"),
        "symptoms": rec.get("symptoms"),
        "image_cues": rec.get("image_cues"),
        "conditions": rec.get("conditions"),
        "treatment": rec.get("treatment"),
        "prevention": rec.get("prevention"),
        "fertilizer_advice": rec.get("fertilizer_advice"),
        "organic_options": rec.get("organic_options"),
        "notes": rec.get("notes"),
        "yield_loss": rec.get("yield_loss"),
        "lookalikes": rec.get("lookalikes"),
    }
    return Document(
        doc_id=rec["id"],
        kind="disease",
        title=title,
        text=flatten_text(searchable),
        payload={"id": rec["id"]},
        boost=1.15,
    )


def _product_document(rec: dict[str, Any]) -> Document:
    title = f"{rec.get('name_th','')} ({rec.get('name_en','')}) - {rec.get('type','')}"
    return Document(
        doc_id=rec["id"],
        kind="product",
        title=title,
        text=flatten_text(
            {
                "group": rec.get("group"),
                "mode": rec.get("mode"),
                "rate_20l": rec.get("rate_20l"),
                "rate_rai": rec.get("rate_rai"),
                "targets": rec.get("targets"),
                "use_stage": rec.get("use_stage"),
                "phi": rec.get("phi_days"),
                "cautions": rec.get("mix_cautions") or rec.get("cautions"),
                "notes": rec.get("notes"),
                "trade": rec.get("trade_examples"),
                "nutrients": rec.get("nutrients"),
            }
        ),
        payload={"id": rec["id"]},
    )


@lru_cache(maxsize=1)
def load_knowledge(data_dir: str | None = None) -> Knowledge:
    """โหลดคลังความรู้ทั้งหมด (แคชไว้ในหน่วยความจำ)"""
    base = Path(data_dir) if data_dir else SETTINGS.data_dir
    diseases: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}
    for name in DISEASE_FILES:
        path = base / "diseases" / f"{name}.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        groups[payload["group_id"]] = payload["group_th"]
        for item in payload["items"]:
            item["group_id"] = payload["group_id"]
            item["group_th"] = payload["group_th"]
            diseases[item["id"]] = item

    pesticide_data = _read_json(base / "pesticides.json")
    fertilizer_data = _read_json(base / "fertilizers.json")
    products: dict[str, dict[str, Any]] = {}
    for item in pesticide_data["items"]:
        item["source"] = "pesticide"
        products[item["id"]] = item
    for item in fertilizer_data["products"]:
        item["source"] = "fertilizer"
        products.setdefault(item["id"], item)

    faq = _read_json(base / "faq.json")["items"]
    crop = _read_json(base / "crop.json")
    sources = _read_json(base / "sources.json")["items"]

    documents: list[Document] = []
    for rec in diseases.values():
        documents.append(_disease_document(rec))
    for rec in products.values():
        documents.append(_product_document(rec))
    for item in faq:
        documents.append(
            Document(
                doc_id=item["id"],
                kind="faq",
                title=item["q"],
                text=f"{item['a']} {' '.join(item.get('tags', []))}",
                payload={"id": item["id"]},
                boost=1.1,
            )
        )
    for key, section in crop.items():
        if key in {"schema_version", "updated"}:
            continue
        documents.append(
            Document(
                doc_id=f"crop_{key}",
                kind="crop",
                title=f"ข้อมูลการปลูกแตงโม: {key}",
                text=flatten_text(section),
                payload={"id": f"crop_{key}"},
            )
        )
    for key in ("programs", "deficiency_quickref", "foliar_rules"):
        if key in fertilizer_data:
            documents.append(
                Document(
                    doc_id=f"fertilizer_{key}",
                    kind="fertilizer_program",
                    title=f"โปรแกรมปุ๋ยและธาตุอาหาร: {key}",
                    text=flatten_text(fertilizer_data[key]),
                    payload={"id": f"fertilizer_{key}"},
                )
            )
    documents.append(
        Document(
            doc_id="pesticide_safety",
            kind="safety",
            title="กฎความปลอดภัยในการใช้สารป้องกันกำจัดศัตรูพืช",
            text=flatten_text(
                {
                    "rules": pesticide_data.get("safety_rules"),
                    "banned": pesticide_data.get("banned_or_restricted_th"),
                    "disclaimer": pesticide_data.get("disclaimer"),
                }
            ),
            payload={"id": "pesticide_safety"},
        )
    )

    return Knowledge(
        diseases=diseases,
        groups=groups,
        products=products,
        fertilizer_data=fertilizer_data,
        pesticide_data=pesticide_data,
        faq=faq,
        crop=crop,
        sources=sources,
        index=BM25Index(documents),
    )
