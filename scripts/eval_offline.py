#!/usr/bin/env python3
"""วัดความแม่นยำของตัววิเคราะห์ภาพโหมดออฟไลน์กับชุดภาพจริง

โหมดออฟไลน์ (app/vision_offline.py) ใช้การวิเคราะห์สีและลักษณะแผล ไม่ใช่โมเดล AI
ความแม่นยำจึงจำกัด สคริปต์นี้มีไว้วัดผลด้วยตัวเลขจริง เพื่อให้การปรับกฎให้คะแนน
อ้างอิงข้อมูลแทนการเดา และเพื่อให้ตรวจสอบได้ว่าการแก้ไขทำให้ดีขึ้นหรือแย่ลง

โครงสร้างโฟลเดอร์ที่ใช้วัด (ชื่อโฟลเดอร์คือคลาสจริงของภาพ):
    <dataset>/<ชื่อคลาส>/*.jpg

ตัวอย่างการใช้งาน:
    # วัดโดยจับคู่ชื่อโฟลเดอร์กับรหัสโรคในคลังความรู้เอง
    python3 scripts/eval_offline.py datasets/watermelon/val

    # กำหนดการจับคู่เองผ่านไฟล์ JSON {"ชื่อโฟลเดอร์": ["รหัสโรคที่ถือว่าถูก", ...]}
    python3 scripts/eval_offline.py datasets/plantdoc/train --mapping eval_map.json

    # แบ่งชุดปรับกฎกับชุดตรวจสอบ เพื่อไม่ให้ตัวเลขเกิดจากการปรับจนเข้ากับภาพชุดเดียว
    python3 scripts/eval_offline.py <dir> --limit 30 --offset 0    # ชุดที่ใช้ปรับ
    python3 scripts/eval_offline.py <dir> --limit 40 --offset 40   # ชุดตรวจสอบ
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision_offline import analyze_offline, extract_features  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPORT_FEATURES = (
    "plant_coverage",
    "pct_whitish",
    "pct_yellow",
    "pct_lesion",
    "lesion_blobs",
    "pct_brown",
)


def load_mapping(path: str | None) -> dict[str, list[str]] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def accepted_ids(folder_name: str, mapping: dict[str, list[str]] | None) -> set[str]:
    """รหัสโรคที่ถือว่าทายถูกสำหรับโฟลเดอร์นี้"""
    if mapping and folder_name in mapping:
        return set(mapping[folder_name])
    # ค่าเริ่มต้น: ใช้ชื่อโฟลเดอร์เป็นรหัสโรคโดยตรง
    return {folder_name}


def evaluate(
    dataset: Path, mapping: dict[str, list[str]] | None, limit: int, offset: int
) -> int:
    folders = sorted(p for p in dataset.iterdir() if p.is_dir())
    if not folders:
        print(f"ไม่พบโฟลเดอร์คลาสใน {dataset}")
        return 1

    print(f"=== วัดผลตัววิเคราะห์ออฟไลน์: {dataset} ===")
    print(f"    ใช้ภาพคลาสละไม่เกิน {limit} ภาพ เริ่มที่ลำดับ {offset}\n")

    overall_top1: list[float] = []
    for folder in folders:
        targets = accepted_ids(folder.name, mapping)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        files = files[offset : offset + limit]
        if not files:
            continue

        rows: list[dict[str, Any]] = []
        top1 = top2 = failed = 0
        confusion: dict[str, int] = {}
        for path in files:
            try:
                raw = path.read_bytes()
                rows.append(extract_features(raw))
                result = analyze_offline(raw)
            except Exception:  # noqa: BLE001 - ภาพเสียหรืออ่านไม่ได้
                failed += 1
                continue
            ids = [c["disease_id"] for c in result["candidates"]]
            if ids:
                confusion[ids[0]] = confusion.get(ids[0], 0) + 1
            if ids and ids[0] in targets:
                top1 += 1
            if any(i in targets for i in ids[:2]):
                top2 += 1

        if not rows:
            continue
        pct1 = top1 / len(rows) * 100
        overall_top1.append(pct1)
        print(f"{folder.name}")
        print(f"    จำนวนภาพ {len(rows)} (อ่านไม่ได้ {failed})")
        print(f"    ทายถูกอันดับหนึ่ง {pct1:.1f}% · ติดสองอันดับแรก {top2 / len(rows) * 100:.1f}%")
        medians = {
            key: round(statistics.median([r[key] for r in rows]), 2) for key in REPORT_FEATURES
        }
        print(f"    ค่ากลางคุณลักษณะ: {json.dumps(medians, ensure_ascii=False)}")
        common = sorted(confusion.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"    ระบบทายเป็น: {', '.join(f'{k} {v} ภาพ' for k, v in common)}\n")

    if overall_top1:
        print(f"เฉลี่ยทุกคลาส: ทายถูกอันดับหนึ่ง {statistics.mean(overall_top1):.1f}%")
        print(
            "\nหมายเหตุ: ตัวเลขนี้คือความแม่นยำของ 'โหมดออฟไลน์' ซึ่งเป็นทางถอยเมื่อไม่มี API key\n"
            "โหมดหลักของระบบคือการวิเคราะห์ด้วยโมเดล Claude ซึ่งแม่นยำกว่ามากและให้เหตุผลประกอบได้"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="วัดความแม่นยำของตัววิเคราะห์ภาพโหมดออฟไลน์",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help="โฟลเดอร์ที่มีโฟลเดอร์ย่อยเป็นคลาสของภาพ")
    parser.add_argument("--mapping", help="ไฟล์ JSON จับคู่ชื่อโฟลเดอร์กับรหัสโรคที่ถือว่าถูก")
    parser.add_argument("--limit", type=int, default=40, help="จำนวนภาพสูงสุดต่อคลาส")
    parser.add_argument("--offset", type=int, default=0, help="เริ่มนับจากภาพลำดับที่เท่าไร")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        print(f"ไม่พบโฟลเดอร์ {dataset}")
        return 1
    return evaluate(dataset, load_mapping(args.mapping), args.limit, args.offset)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nยกเลิกการทำงาน")
        sys.exit(130)
