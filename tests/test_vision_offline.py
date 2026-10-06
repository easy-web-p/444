"""ทดสอบตัววิเคราะห์ภาพแบบออฟไลน์"""

from __future__ import annotations

from app.vision_offline import CONFIDENCE_CAP, analyze_offline, extract_features


def test_features_detect_lesion_blobs(leaf_with_spots):
    features = extract_features(leaf_with_spots)
    assert features["pct_green"] > 50
    assert features["lesion_blobs"] >= 5, "ควรตรวจพบจุดแผลที่แยกกันได้"
    assert features["pct_lesion"] > 0.5


def test_healthy_leaf_has_minimal_damage(healthy_leaf):
    features = extract_features(healthy_leaf)
    assert features["pct_green"] > 80
    assert features["pct_lesion"] < 2


def test_spotty_leaf_suggests_leaf_spot_group(leaf_with_spots):
    result = analyze_offline(leaf_with_spots)
    ids = [c["disease_id"] for c in result["candidates"]]
    assert "anthracnose" in ids or "alternaria_blight" in ids


def test_white_coating_suggests_powdery_mildew(white_coated_leaf):
    result = analyze_offline(white_coated_leaf)
    assert result["candidates"]
    assert result["candidates"][0]["disease_id"] == "powdery_mildew"


def test_confidence_is_capped_in_offline_mode(leaf_with_spots):
    """โหมดออฟไลน์ต้องไม่อ้างความมั่นใจสูงเกินจริง"""
    result = analyze_offline(leaf_with_spots)
    for candidate in result["candidates"]:
        assert candidate["confidence"] <= CONFIDENCE_CAP


def test_offline_result_has_required_shape(leaf_with_spots):
    result = analyze_offline(leaf_with_spots)
    for key in (
        "is_plant_image",
        "plant_part",
        "image_quality",
        "observations",
        "candidates",
        "severity",
        "immediate_actions",
        "need_more_checks",
        "summary_th",
    ):
        assert key in result
    assert result["_meta"]["engine"] == "offline_heuristic"
    assert "ออฟไลน์" in result["summary_th"]


def test_soil_background_is_not_counted_as_lesions(leaf_on_soil_background):
    """ดินและเงาในพื้นหลังต้องไม่ถูกนับเป็นจุดแผล

    ก่อนแก้ ภาพถ่ายจากแปลงจริงถูกนับจุดแผลจากเงาระหว่างก้อนดินนับร้อยจุด
    ทำให้ภาพราแป้งถูกจัดเป็นโรคใบจุดผิด
    """
    features = extract_features(leaf_on_soil_background)
    assert features["plant_coverage"] < 60, "พื้นหลังดินต้องไม่ถูกนับเป็นพื้นที่พืช"
    assert features["pct_lesion"] < 8, "พื้นที่แผลต้องคิดเฉพาะในบริเวณใบเท่านั้น"


def test_white_patches_outrank_leaf_spot_on_field_like_image(leaf_on_soil_background):
    """ภาพใบมีคราบขาวบนพื้นหลังดิน ต้องได้ราแป้งเป็นอันดับหนึ่ง"""
    result = analyze_offline(leaf_on_soil_background)
    assert result["candidates"], "ต้องเสนอสาเหตุอย่างน้อยหนึ่งรายการ"
    assert result["candidates"][0]["disease_id"] == "powdery_mildew"


def test_percentages_are_relative_to_plant_area(leaf_with_spots):
    """เปอร์เซ็นต์ต้องคิดเทียบกับพื้นที่พืช ไม่ใช่ทั้งภาพ"""
    features = extract_features(leaf_with_spots)
    assert "plant_coverage" in features
    assert 0 <= features["pct_green"] <= 100
    assert features["pct_green"] > 70, "ภาพใบเต็มกรอบ พื้นที่เขียวเทียบกับพื้นที่ใบต้องสูง"
