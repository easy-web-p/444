#!/usr/bin/env python3
"""วิเคราะห์ภาพโรคแตงโมจากบรรทัดคำสั่ง โดยไม่ต้องเปิดหน้าเว็บ

เหมาะสำหรับทดสอบว่าระบบทำงานถูกต้องหลังใส่ API key แล้ว และสำหรับวิเคราะห์ภาพ
หลายไฟล์พร้อมกันเพื่อเก็บผลเป็นไฟล์

ตัวอย่างการใช้งาน:
    python3 scripts/diagnose_image.py ภาพใบแตงโม.jpg
    python3 scripts/diagnose_image.py ภาพ.jpg --context "อายุ 45 วัน ฝนตก 3 วันติด ใบล่างเป็นก่อน"
    python3 scripts/diagnose_image.py *.jpg --json ผลวิเคราะห์.json
    python3 scripts/diagnose_image.py ภาพ.jpg --full      # แสดงแผนการจัดการแบบเต็ม

ก่อนใช้งานให้ตั้งค่า API key ก่อน มิฉะนั้นจะทำงานในโหมดออฟไลน์ซึ่งแม่นยำน้อยกว่ามาก:
    cp .env.example .env     แล้วใส่ค่า ANTHROPIC_API_KEY ในไฟล์
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.diagnose import ImageError, diagnose  # noqa: E402
from app.knowledge import load_knowledge  # noqa: E402

LINE = "─" * 68


def bar(percent: int, width: int = 24) -> str:
    filled = max(0, min(width, round(percent / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def print_report(path: Path, result: dict[str, Any], show_full: bool) -> None:
    print(f"\n{LINE}\nไฟล์: {path.name}")
    engine = (
        f"โมเดล AI ({result['meta'].get('model', SETTINGS.model)})"
        if result["engine"] == "claude_vision"
        else "โหมดออฟไลน์ (วิเคราะห์สีและลักษณะแผล ความแม่นยำจำกัด)"
    )
    print(f"วิเคราะห์ด้วย: {engine}")
    if result.get("plant_part"):
        print(f"ส่วนที่วิเคราะห์: {result['plant_part']}")
    print(LINE)

    if not result.get("is_plant_image", True):
        print("\n[!] ภาพนี้อาจไม่ใช่ภาพพืชหรือส่วนของพืช")

    quality = result.get("image_quality") or {}
    if quality.get("usable") is False:
        print("\n[!] คุณภาพภาพยังไม่เหมาะกับการวิเคราะห์")
        for issue in quality.get("issues", []):
            print(f"    - {issue}")
        for advice in quality.get("advice", []):
            print(f"    แนะนำ: {advice}")

    if result.get("summary_th"):
        print(f"\nสรุปผล\n  {result['summary_th']}")

    if result.get("observations"):
        print("\nสิ่งที่เห็นในภาพ")
        for item in result["observations"]:
            print(f"  - {item}")

    print("\nสาเหตุที่เป็นไปได้ (ตัวเลขคือความมั่นใจ ไม่ใช่ความรุนแรง)")
    for i, cand in enumerate(result.get("candidates", []), 1):
        pct = cand["confidence"]
        kind = ""
        if cand.get("is_infectious") is False:
            kind = "  [ไม่ใช่โรคติดเชื้อ]"
        elif cand.get("is_infectious") is True:
            kind = "  [โรคติดเชื้อ]"
        print(f"  {i}. {cand['name_th']}{kind}")
        print(f"     {bar(pct)} {pct}%")
        for ev in cand.get("evidence", [])[:3]:
            print(f"     วิเคราะห์จาก: {ev}")
        for against in cand.get("against", [])[:2]:
            print(f"     ยังไม่ยืนยัน: {against}")

    severity = result.get("severity") or {}
    if severity.get("level"):
        print(
            f"\nความรุนแรง: {severity['level']} "
            f"(พื้นที่เสียหายประมาณ {severity.get('affected_area_percent', '-')}%)"
        )
        if severity.get("spread_risk"):
            print(f"ความเสี่ยงการลุกลาม: {severity['spread_risk']}")

    for note in (result.get("safety") or {}).get("guidance", []):
        print(f"\n[!] {note}")

    if result.get("immediate_actions"):
        print("\nสิ่งที่ควรทำทันที")
        for item in result["immediate_actions"]:
            print(f"  - {item}")

    if result.get("need_more_checks"):
        print("\nต้องไปตรวจเพิ่มในแปลงเพื่อยืนยัน")
        for item in result["need_more_checks"]:
            print(f"  - {item}")

    plan = result.get("treatment")
    if plan:
        print(f"\n{LINE}\nแผนการจัดการ: {plan['name_th']}")
        if plan.get("chemical"):
            print("\nสารที่ใช้ได้ (อ่านฉลากและใช้อัตราตามฉลากเสมอ)")
            for item in plan["chemical"][: None if show_full else 3]:
                phi = (
                    f" · เก็บเกี่ยวได้หลังพ่น {item['phi_days']} วัน"
                    if item.get("phi_days") is not None
                    else ""
                )
                print(f"  - {item['name_th']} ({item.get('group', '')})")
                print(f"    อัตรา {item['rate']} · ทุก {item.get('interval', '-')}{phi}")
                if show_full and item.get("note"):
                    print(f"    หมายเหตุ: {item['note']}")
        if plan.get("biological"):
            print("\nชีวภัณฑ์และทางเลือกที่ปลอดภัยกว่า")
            for item in plan["biological"][: None if show_full else 2]:
                print(f"  - {item['name_th']} อัตรา {item['rate']}")
        if show_full:
            for key, title in (
                ("urgent", "ทำทันที"),
                ("cultural", "การจัดการแปลง"),
                ("fertilizer_advice", "คำแนะนำปุ๋ย"),
                ("prevention", "การป้องกันรอบปลูกถัดไป"),
            ):
                if plan.get(key):
                    print(f"\n{title}")
                    for item in plan[key]:
                        print(f"  - {item}")
            if plan.get("lookalikes"):
                print("\nโรคที่อาการคล้ายกันและจุดแยก")
                for look in plan["lookalikes"]:
                    print(f"  - {look['name_th']}: {look['key_difference']}")

    safety = result.get("safety") or {}
    if safety.get("max_phi_days") is not None:
        print(
            f"\n[!] ระยะเก็บเกี่ยวปลอดภัยที่นานที่สุดในแผนนี้คือ {safety['max_phi_days']} วัน"
        )
    if safety.get("bee_risky_products"):
        print(
            "[!] สารที่เป็นพิษต่อผึ้งสูงในแผนนี้: "
            + ", ".join(safety["bee_risky_products"])
            + " — ห้ามพ่นช่วงดอกบาน"
        )

    print(f"\n{result.get('disclaimer', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="วิเคราะห์ภาพโรคแตงโมจากบรรทัดคำสั่ง",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("images", nargs="+", help="ไฟล์ภาพที่ต้องการวิเคราะห์")
    parser.add_argument("--context", default="", help="ข้อมูลประกอบ เช่น อายุต้น สภาพอากาศ")
    parser.add_argument("--full", action="store_true", help="แสดงแผนการจัดการแบบเต็ม")
    parser.add_argument("--json", dest="json_out", help="บันทึกผลทั้งหมดเป็นไฟล์ JSON")
    args = parser.parse_args()

    if not SETTINGS.ai_enabled:
        print(
            "[!] ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY จึงใช้โหมดออฟไลน์ซึ่งแม่นยำน้อยกว่ามาก\n"
            "    วิธีเปิดใช้งานเต็มรูปแบบ: cp .env.example .env แล้วใส่ค่า ANTHROPIC_API_KEY"
        )

    kb = load_knowledge()
    results: list[dict[str, Any]] = []
    for name in args.images:
        path = Path(name)
        if not path.is_file():
            print(f"[!] ไม่พบไฟล์ {path}")
            continue
        started = time.time()
        try:
            result = diagnose(path.read_bytes(), kb, "image/jpeg", args.context)
        except ImageError as exc:
            print(f"[!] {path.name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[!] {path.name}: วิเคราะห์ไม่สำเร็จ: {exc}")
            continue
        print_report(path, result, args.full)
        print(f"ใช้เวลา {time.time() - started:.1f} วินาที")
        results.append({"file": str(path), "result": result})

    if args.json_out and results:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nบันทึกผลทั้งหมดไว้ที่ {args.json_out}")
    return 0 if results else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nยกเลิกการทำงาน")
        sys.exit(130)
