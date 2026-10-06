"""ทดสอบว่าคลังความรู้ผ่านการตรวจสอบของ scripts/validate_data.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_knowledge_base_passes_validation():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_data.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"คลังความรู้ไม่ผ่านการตรวจสอบ:\n{result.stdout}\n{result.stderr}"
    assert "ข้อผิดพลาด 0 รายการ" in result.stdout
