#!/usr/bin/env python3
"""ตรวจสอบความถูกต้องของคลังความรู้ทั้งหมด

ตรวจ:
- ไฟล์ JSON อ่านได้และมีฟิลด์ที่จำเป็นครบ
- รหัส id ไม่ซ้ำกัน
- การอ้างอิงข้ามไฟล์ถูกต้อง (ยา/ปุ๋ยที่โรคอ้างถึงต้องมีจริง, โรคที่ยาระบุเป็นเป้าหมายต้องมีจริง,
  โรคที่ระบุว่าอาการคล้ายกันต้องมีจริง, แหล่งอ้างอิงต้องมีจริง)
- ฟิลด์ที่สำคัญต่อความปลอดภัยไม่ว่าง เช่น อัตราการใช้และระยะเก็บเกี่ยวปลอดภัย

ใช้งาน:  python3 scripts/validate_data.py [--data-dir data]
คืนค่า exit code 1 เมื่อพบข้อผิดพลาด เหมาะกับการใช้ใน CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DISEASE_FILES = ("fungal", "oomycete", "bacterial", "viral", "nematode", "abiotic", "pests")

REQUIRED_DISEASE_FIELDS = (
    "id",
    "name_th",
    "pathogen_type",
    "severity",
    "affected_parts",
    "symptoms",
    "image_cues",
    "treatment",
    "prevention",
)
REQUIRED_PESTICIDE_FIELDS = ("id", "name_th", "name_en", "type", "group", "rate_20l", "targets")
VALID_PATHOGEN_TYPES = {
    "fungus",
    "oomycete",
    "bacterium",
    "virus",
    "nematode",
    "abiotic",
    "insect",
    "mite",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> int:
        for item in self.warnings:
            print(f"  [เตือน]  {item}")
        for item in self.errors:
            print(f"  [ผิดพลาด] {item}")
        print()
        print(f"สรุป: ข้อผิดพลาด {len(self.errors)} รายการ, คำเตือน {len(self.warnings)} รายการ")
        return 1 if self.errors else 0


def load(path: Path, report: Report) -> dict[str, Any] | None:
    if not path.is_file():
        report.error(f"ไม่พบไฟล์ {path}")
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        report.error(f"ไฟล์ {path} ไม่ใช่ JSON ที่ถูกต้อง: {exc}")
        return None


def validate(data_dir: Path) -> int:
    report = Report()
    diseases: dict[str, dict[str, Any]] = {}

    # ---------- ไฟล์โรค ----------
    for name in DISEASE_FILES:
        payload = load(data_dir / "diseases" / f"{name}.json", report)
        if payload is None:
            continue
        if "group_id" not in payload or "items" not in payload:
            report.error(f"diseases/{name}.json ต้องมีฟิลด์ group_id และ items")
            continue
        for item in payload["items"]:
            disease_id = item.get("id", "")
            if not disease_id:
                report.error(f"diseases/{name}.json มีรายการที่ไม่มี id")
                continue
            if disease_id in diseases:
                report.error(f"รหัสโรคซ้ำกัน: {disease_id}")
            for field in REQUIRED_DISEASE_FIELDS:
                if field not in item or item[field] in ("", None, [], {}):
                    report.error(f"{disease_id}: ไม่มีฟิลด์ที่จำเป็น '{field}'")
            if item.get("pathogen_type") not in VALID_PATHOGEN_TYPES:
                report.error(
                    f"{disease_id}: pathogen_type '{item.get('pathogen_type')}' ไม่อยู่ในรายการที่กำหนด"
                )
            if not item.get("image_cues"):
                report.warn(f"{disease_id}: ไม่มี image_cues ทำให้ AI วิเคราะห์จากภาพได้ยาก")
            item["_file"] = name
            diseases[disease_id] = item

    # ---------- ยาและสาร ----------
    pesticides = load(data_dir / "pesticides.json", report) or {"items": []}
    products: dict[str, dict[str, Any]] = {}
    for item in pesticides.get("items", []):
        pid = item.get("id", "")
        if not pid:
            report.error("pesticides.json มีรายการที่ไม่มี id")
            continue
        if pid in products:
            report.error(f"รหัสสารซ้ำกัน: {pid}")
        for field in REQUIRED_PESTICIDE_FIELDS:
            if field not in item or item[field] in ("", None, []):
                report.error(f"สาร {pid}: ไม่มีฟิลด์ที่จำเป็น '{field}'")
        if item.get("type") != "adjuvant" and item.get("phi_days") is None:
            report.warn(f"สาร {pid}: ไม่ได้ระบุระยะเก็บเกี่ยวปลอดภัย (phi_days)")
        products[pid] = item

    # ---------- ปุ๋ย ----------
    fertilizers = load(data_dir / "fertilizers.json", report) or {}
    for item in fertilizers.get("products", []):
        pid = item.get("id", "")
        if not pid:
            report.error("fertilizers.json มีรายการที่ไม่มี id")
            continue
        if pid in products:
            report.warn(f"รหัส {pid} มีทั้งใน pesticides.json และ fertilizers.json")
        products.setdefault(pid, item)

    # ---------- แหล่งอ้างอิง ----------
    sources = (load(data_dir / "sources.json", report) or {}).get("items", {})

    # ---------- การอ้างอิงข้ามไฟล์ ----------
    for disease_id, item in diseases.items():
        treatment = item.get("treatment", {}) or {}
        for key in ("chemical", "biological"):
            for entry in treatment.get(key, []) or []:
                ref = entry.get("ref")
                if ref not in products:
                    report.error(f"{disease_id}: อ้างถึงผลิตภัณฑ์ '{ref}' ที่ไม่มีในคลังความรู้")
                if not entry.get("rate"):
                    report.error(f"{disease_id}: รายการ '{ref}' ไม่ได้ระบุอัตราการใช้")
        for look in item.get("lookalikes", []) or []:
            if "id" in look and look["id"] not in diseases:
                report.error(f"{disease_id}: lookalike '{look['id']}' ไม่มีในคลังความรู้")
            if not look.get("key_difference"):
                report.warn(f"{disease_id}: lookalike ไม่มีคำอธิบายจุดแยก")
        for ref in item.get("refs", []) or []:
            if ref not in sources:
                report.error(f"{disease_id}: อ้างแหล่งข้อมูล '{ref}' ที่ไม่มีใน sources.json")

    for pid, item in products.items():
        for target in item.get("targets", []) or []:
            if target not in diseases:
                report.error(f"สาร {pid}: ระบุเป้าหมาย '{target}' ที่ไม่มีในคลังความรู้")
        for ref in item.get("refs", []) or []:
            if ref not in sources:
                report.error(f"สาร {pid}: อ้างแหล่งข้อมูล '{ref}' ที่ไม่มีใน sources.json")

    # ---------- FAQ และข้อมูลพืช ----------
    faq = (load(data_dir / "faq.json", report) or {}).get("items", [])
    faq_ids: set[str] = set()
    for item in faq:
        fid = item.get("id", "")
        if fid in faq_ids:
            report.error(f"รหัส FAQ ซ้ำกัน: {fid}")
        faq_ids.add(fid)
        if not item.get("q") or not item.get("a"):
            report.error(f"FAQ {fid}: ต้องมีทั้งคำถามและคำตอบ")
        for rel in item.get("related", []) or []:
            if rel not in diseases:
                report.error(f"FAQ {fid}: อ้างถึงโรค '{rel}' ที่ไม่มีในคลังความรู้")

    crop = load(data_dir / "crop.json", report) or {}
    for entry in crop.get("quick_triage", []) or []:
        if not entry.get("if") or not entry.get("then"):
            report.error("crop.json: quick_triage ต้องมีทั้งเงื่อนไขและคำแนะนำ")

    fert_quickref = fertilizers.get("deficiency_quickref", []) or []
    for entry in fert_quickref:
        if entry.get("disease_id") and entry["disease_id"] not in diseases:
            report.error(
                f"fertilizers.json: deficiency_quickref อ้างถึงโรค '{entry['disease_id']}' ที่ไม่มีอยู่"
            )

    unused_sources = set(sources) - {
        ref
        for item in list(diseases.values()) + list(products.values())
        for ref in item.get("refs", []) or []
    }
    for key in sorted(unused_sources):
        report.warn(f"แหล่งอ้างอิง '{key}' ไม่ได้ถูกใช้ที่ใดเลย")

    print("=== ตรวจสอบคลังความรู้ ===")
    print(f"  โรคและอาการผิดปกติ : {len(diseases)} รายการ")
    print(f"  ยาและปุ๋ย          : {len(products)} รายการ")
    print(f"  คำถามที่พบบ่อย      : {len(faq)} ข้อ")
    print(f"  แหล่งอ้างอิง        : {len(sources)} รายการ")
    print()
    return report.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description="ตรวจสอบความถูกต้องของคลังความรู้")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="โฟลเดอร์ข้อมูล (ค่าเริ่มต้น: data/)",
    )
    args = parser.parse_args()
    return validate(Path(args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
