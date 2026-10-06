"""ทดสอบการค้นคืนภาษาไทย (BM25 แบบ character n-gram)"""

from __future__ import annotations

import pytest

from app.retrieval import BM25Index, Document, tokenize


def test_tokenize_handles_thai_and_ascii():
    tokens = tokenize("ราน้ำค้าง downy mildew 25%")
    assert "downy" in tokens and "mildew" in tokens and "25" in tokens
    assert any(len(t) == 2 for t in tokens), "ต้องมี n-gram ภาษาไทย"
    assert tokenize("") == []


def test_bm25_ranks_relevant_document_first():
    docs = [
        Document("a", "disease", "ราน้ำค้าง", "ปื้นเหลืองรูปเหลี่ยมบนใบและขุยราสีเทาอมม่วงใต้ใบ"),
        Document("b", "disease", "ราแป้ง", "ผงสีขาวคล้ายแป้งบนผิวใบด้านบน เช็ดออกได้"),
        Document("c", "disease", "รากปม", "รากมีปุ่มปมบวมและต้นแคระ"),
    ]
    index = BM25Index(docs)
    top = index.search("ใต้ใบมีขุยสีเทาและปื้นเหลือง", limit=1)
    assert top and top[0][0].doc_id == "a"


def test_search_can_filter_by_kind():
    docs = [
        Document("d1", "disease", "โรคใบจุด", "จุดสีน้ำตาลบนใบ"),
        Document("p1", "product", "แมนโคเซบ", "สารป้องกันโรคพืช จุดสีน้ำตาลบนใบ"),
    ]
    index = BM25Index(docs)
    hits = index.search("จุดสีน้ำตาล", kinds=["product"])
    assert [doc.doc_id for doc, _ in hits] == ["p1"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("โรคราน้ำค้างรักษาอย่างไร", "downy_mildew"),
        ("มียางสีน้ำตาลไหลจากเถา", "gummy_stem_blight"),
        ("รากมีปุ่มปมและต้นแคระ", "root_knot_nematode"),
        ("ปลายผลเน่าดำ", "blossom_end_rot"),
        ("ผงขาวบนใบเหมือนแป้ง", "powdery_mildew"),
        ("ใบด่างและใบอ่อนหงิกเสียรูป", "watermelon_mosaic_virus"),
    ],
)
def test_real_symptom_queries_hit_right_disease(kb, query, expected):
    hits = kb.index.search(query, limit=3, kinds=["disease"])
    assert expected in [doc.doc_id for doc, _ in hits], f"ค้นหา '{query}' ไม่เจอ {expected}"


def test_exact_product_name_outranks_combination_product(kb):
    """ถามถึงสารเดี่ยวต้องได้สารเดี่ยวก่อนสูตรผสมที่มีชื่อนั้นอยู่ด้วย"""
    hits = kb.index.search("อัตราใช้แมนโคเซบ", limit=3, kinds=["product"])
    assert hits[0][0].doc_id == "mancozeb"
