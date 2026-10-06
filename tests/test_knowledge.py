"""ทดสอบคลังความรู้: การโหลด ความครบถ้วน และการประกอบแผนการรักษา"""

from __future__ import annotations

import pytest


def test_knowledge_loads_expected_volume(kb):
    stats = kb.stats()
    assert stats["diseases"] >= 40, "ควรมีข้อมูลโรคและอาการผิดปกติอย่างน้อย 40 รายการ"
    assert stats["products"] >= 80, "ควรมีข้อมูลยาและปุ๋ยอย่างน้อย 80 รายการ"
    assert stats["faq"] >= 20
    assert stats["sources"] >= 20


def test_every_group_present(kb):
    expected = {"fungal", "oomycete", "bacterial", "viral", "nematode", "abiotic", "pest"}
    assert expected.issubset(set(kb.groups))


@pytest.mark.parametrize(
    "disease_id",
    [
        "anthracnose",
        "downy_mildew",
        "gummy_stem_blight",
        "fusarium_wilt",
        "bacterial_fruit_blotch",
        "watermelon_mosaic_virus",
        "root_knot_nematode",
        "blossom_end_rot",
        "aphid",
    ],
)
def test_key_diseases_have_complete_plan(kb, disease_id):
    """โรคสำคัญต้องมีข้อมูลครบพอให้เกษตรกรนำไปใช้ได้จริง"""
    plan = kb.treatment_plan(disease_id)
    assert plan is not None
    assert plan["name_th"]
    assert plan["symptoms"], "ต้องมีคำอธิบายอาการ"
    assert plan["urgent"], "ต้องบอกสิ่งที่ต้องทำทันที"
    assert plan["prevention"], "ต้องมีคำแนะนำการป้องกัน"
    assert plan["chemical"] or plan["biological"], "ต้องมีคำแนะนำสารอย่างน้อยหนึ่งกลุ่ม"


def test_all_chemical_entries_have_rate_and_known_product(kb):
    """ทุกคำแนะนำสารต้องมีอัตราการใช้และต้องชี้ไปยังผลิตภัณฑ์ที่มีอยู่จริง"""
    for disease_id in kb.diseases:
        plan = kb.treatment_plan(disease_id)
        for entry in plan["chemical"] + plan["biological"]:
            assert entry["rate"], f"{disease_id}: {entry['ref']} ไม่มีอัตราการใช้"
            assert kb.product(entry["ref"]) is not None, f"{disease_id}: ไม่รู้จัก {entry['ref']}"


def test_non_infectious_entries_flagged(kb):
    """อาการขาดธาตุอาหารและความเสียหายจากแมลงต้องไม่ถูกจัดเป็นโรคติดเชื้อ"""
    for disease_id in ("blossom_end_rot", "nitrogen_deficiency", "aphid", "spider_mite"):
        assert kb.treatment_plan(disease_id)["is_infectious"] is False
    for disease_id in ("anthracnose", "downy_mildew", "bacterial_fruit_blotch"):
        assert kb.treatment_plan(disease_id)["is_infectious"] is True


def test_lookalikes_resolve_to_real_entries(kb):
    for rec in kb.diseases.values():
        for look in rec.get("lookalikes", []):
            if "id" in look:
                assert kb.disease(look["id"]) is not None
                assert look["key_difference"]


def test_resolve_disease_by_various_names(kb):
    assert kb.resolve_disease("ราน้ำค้าง")["id"] == "downy_mildew"
    assert kb.resolve_disease("Anthracnose")["id"] == "anthracnose"
    assert kb.resolve_disease("โรคยางไหล (เถาแห้ง/ลำต้นไหม้)")["id"] == "gummy_stem_blight"
    assert kb.resolve_disease("") is None


def test_disease_catalog_lists_every_id(kb):
    catalog = kb.disease_catalog_for_prompt()
    for disease_id in kb.diseases:
        assert f"id={disease_id}" in catalog


def test_bee_toxic_products_are_marked(kb):
    """สารที่เป็นพิษต่อผึ้งสูงต้องถูกระบุไว้ เพราะแตงโมต้องพึ่งผึ้งผสมเกสร"""
    imidacloprid = kb.product("imidacloprid")
    assert "สูง" in imidacloprid["bee_toxicity"]
    assert any("ดอกบาน" in caution for caution in imidacloprid["mix_cautions"])


def test_phi_present_for_chemical_products(kb):
    """สารเคมีต้องระบุระยะเก็บเกี่ยวปลอดภัย เพื่อป้องกันสารตกค้างเกินมาตรฐาน"""
    for rec in kb.pesticide_data["items"]:
        if rec["type"] in {"adjuvant"}:
            continue
        assert rec.get("phi_days") is not None, f"{rec['id']} ไม่ได้ระบุ phi_days"
