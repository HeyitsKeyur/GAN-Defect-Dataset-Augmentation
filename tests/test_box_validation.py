from src.data.box_validation import validate_xyxy, validate_yolo_normalized


def test_valid_box() -> None:
    issues = validate_xyxy(10, 10, 20, 30, image_width=100, image_height=100)
    assert issues == []


def test_zero_area_box() -> None:
    issues = validate_xyxy(10, 10, 10, 30, image_width=100, image_height=100)
    codes = {issue.code for issue in issues}
    assert "zero_area" in codes


def test_negative_extent() -> None:
    issues = validate_xyxy(20, 10, 10, 30, image_width=100, image_height=100)
    codes = {issue.code for issue in issues}
    assert "negative_extent" in codes
    assert "zero_area" in codes


def test_out_of_bounds_origin() -> None:
    issues = validate_xyxy(-5, 0, 10, 10, image_width=100, image_height=100)
    assert any(issue.code == "out_of_bounds" for issue in issues)


def test_exceeds_image() -> None:
    issues = validate_xyxy(90, 90, 120, 110, image_width=100, image_height=100)
    assert any(issue.code == "exceeds_image" for issue in issues)


def test_rounding_tolerance_not_flagged() -> None:
    issues = validate_xyxy(0.2, 0.2, 100.4, 99.6, image_width=100, image_height=100)
    assert issues == []


def test_invalid_class_id() -> None:
    issues = validate_yolo_normalized("abc", 0.5, 0.5, 0.1, 0.1, allowed_class_ids={0, 1, 2})
    assert any(issue.code == "invalid_class_id" for issue in issues)


def test_disallowed_class_id() -> None:
    issues = validate_yolo_normalized(9, 0.5, 0.5, 0.1, 0.1, allowed_class_ids={0, 1, 2})
    assert any(issue.code == "invalid_class_id" for issue in issues)


def test_yolo_normalized_out_of_range() -> None:
    issues = validate_yolo_normalized(1, 1.2, 0.5, 0.1, 0.1)
    assert any(issue.code == "invalid_normalized_coordinate" for issue in issues)
