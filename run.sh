#!/usr/bin/env bash
# สคริปต์เริ่มต้นใช้งานแบบเร็ว — ติดตั้ง dependencies แล้วรันเซิร์ฟเวอร์
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=${VENV:-.venv}

if [ ! -d "$VENV" ]; then
  echo "==> สร้าง virtual environment ที่ $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> ติดตั้ง dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f .env ]; then
  echo "==> ไม่พบไฟล์ .env — คัดลอกจาก .env.example (ระบบจะรันในโหมดออฟไลน์จนกว่าจะใส่ ANTHROPIC_API_KEY)"
  cp .env.example .env
fi

HOST=${MELON_HOST:-0.0.0.0}
PORT=${MELON_PORT:-8000}
echo "==> เปิดเซิร์ฟเวอร์ที่ http://localhost:${PORT}"
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
