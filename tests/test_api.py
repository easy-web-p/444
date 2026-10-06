"""ทดสอบ API ทั้งหมดผ่าน TestClient (ทำงานในโหมดออฟไลน์ ไม่เรียก API ภายนอก)"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["knowledge"]["diseases"] >= 40
    assert body["mode"] in {"claude", "offline"}


def test_index_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "วินิจฉัยโรคแตงโม" in res.text


def test_static_assets_served(client):
    for path in ("/static/styles.css", "/static/app.js"):
        assert client.get(path).status_code == 200


def test_disease_list_and_filter(client):
    res = client.get("/api/diseases")
    assert res.status_code == 200
    total = res.json()["count"]
    assert total >= 40
    viral = client.get("/api/diseases", params={"group": "viral"}).json()
    assert 0 < viral["count"] < total
    assert all(item["group_id"] == "viral" for item in viral["items"])


def test_disease_detail_contains_actionable_plan(client):
    res = client.get("/api/diseases/downy_mildew")
    assert res.status_code == 200
    body = res.json()
    assert body["name_th"]
    assert body["chemical"], "ต้องมีคำแนะนำสารเคมี"
    assert all(entry["rate"] for entry in body["chemical"])
    assert body["refs"], "ต้องมีแหล่งอ้างอิง"


def test_disease_detail_404(client):
    assert client.get("/api/diseases/not_a_real_disease").status_code == 404


def test_pesticide_endpoints(client):
    res = client.get("/api/pesticides", params={"type": "biological", "organic": "true"})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] > 0
    assert all(item["organic_ok"] for item in body["items"])
    detail = client.get("/api/pesticides/trichoderma").json()
    assert detail["name_th"]
    assert detail["target_details"], "ต้องบอกว่าใช้ควบคุมโรคอะไรได้บ้าง"


def test_pesticides_filtered_by_target(client):
    body = client.get("/api/pesticides", params={"target": "downy_mildew"}).json()
    assert body["count"] > 0
    assert all("downy_mildew" in item["targets"] for item in body["items"])


def test_fertilizer_program_endpoint(client):
    body = client.get("/api/fertilizers").json()
    assert len(body["programs"]["stages"]) >= 4
    assert body["deficiency_quickref"]


def test_search_endpoint(client):
    body = client.get("/api/search", params={"q": "ยางไหลที่เถา", "limit": 3}).json()
    assert body["count"] > 0
    assert "gummy_stem_blight" in [item["id"] for item in body["items"]]


def test_chat_offline_returns_knowledge(client):
    res = client.post("/api/chat", json={"message": "ใบมีผงขาวเหมือนแป้ง ใช้ยาอะไร"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"]
    assert body["sources"]
    if not body["ai_enabled"]:
        assert body["engine"] == "offline_retrieval"
        assert "ราแป้ง" in body["answer"]


def test_chat_stream_emits_sse_events(client):
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "โรคราน้ำค้างคืออะไร"}
    ) as res:
        assert res.status_code == 200
        payloads = []
        for line in res.iter_lines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))
    types = [p["type"] for p in payloads]
    assert types[0] == "meta"
    assert "done" in types
    assert any(p["type"] == "delta" for p in payloads)


def test_chat_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_diagnose_returns_plan(client, leaf_with_spots):
    res = client.post(
        "/api/diagnose",
        files={"image": ("leaf.jpg", leaf_with_spots, "image/jpeg")},
        data={"context": "แตงโมอายุ 45 วัน ฝนตกติดกัน 3 วัน"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["candidates"], "ต้องเสนอสาเหตุที่เป็นไปได้อย่างน้อยหนึ่งรายการ"
    top = body["candidates"][0]
    assert 0 <= top["confidence"] <= 100
    assert top["evidence"], "ต้องบอกว่าวิเคราะห์จากหลักฐานอะไร"
    assert body["disclaimer"]
    assert body["safety"]["rules"]
    if top["in_knowledge_base"]:
        assert body["treatment"] is not None
        assert body["treatment"]["chemical"] or body["treatment"]["biological"]


def test_diagnose_rejects_non_image(client):
    res = client.post(
        "/api/diagnose",
        files={"image": ("note.txt", b"this is not an image", "text/plain")},
    )
    assert res.status_code == 400


def test_diagnose_rejects_corrupt_image(client):
    res = client.post(
        "/api/diagnose",
        files={"image": ("broken.jpg", b"\xff\xd8\xff\xe0 not really a jpeg", "image/jpeg")},
    )
    assert res.status_code == 400
