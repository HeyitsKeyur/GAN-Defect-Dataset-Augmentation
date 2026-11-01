from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.annotation_parsers import parse_labelme_json
from src.data.dataset_paths import index_by_stem
from src.data.image_utils import infer_group_id, inspect_image


def test_stem_index_matches_image_and_json(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "json"
    images.mkdir()
    labels.mkdir()
    (images / "sample_a.jpg").write_bytes(b"not-an-image")
    (labels / "sample_a.json").write_text("{}", encoding="utf-8")
    (labels / "orphan.json").write_text("{}", encoding="utf-8")
    image_index = index_by_stem(list(images.iterdir()))
    json_index = index_by_stem(list(labels.iterdir()))
    missing_json = set(image_index) - set(json_index)
    missing_images = set(json_index) - set(image_index)
    assert missing_json == set()
    assert missing_images == {"orphan"}


def test_inspect_unreadable_image(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not jpeg data")
    info = inspect_image(path)
    assert info.readable is False
    assert info.error


def test_inspect_readable_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "ok.jpg"
    Image.new("RGB", (32, 16), color=(12, 34, 56)).save(path, format="JPEG")
    info = inspect_image(path)
    assert info.readable is True
    assert info.width == 32
    assert info.height == 16
    assert info.format_name == "JPEG"
    assert info.md5


def test_json_dimension_mismatch_is_detectable(tmp_path: Path) -> None:
    json_path = tmp_path / "ok.json"
    json_path.write_text(
        '{"shapes":[{"label":"spot","points":[[1,1],[2,1],[2,2],[1,2]],"shape_type":"rectangle"}],'
        '"imageWidth":99,"imageHeight":99}',
        encoding="utf-8",
    )
    image_path = tmp_path / "ok.jpg"
    Image.new("RGB", (32, 16), color=0).save(image_path, format="JPEG")
    parsed = parse_labelme_json(json_path)
    info = inspect_image(image_path)
    assert parsed.recorded_width != info.width
    assert parsed.recorded_height != info.height


def test_filename_group_for_tiled_parts() -> None:
    group_a, evidence_a = infer_group_id("1_2_000001_0810104210546_NG_part0")
    group_b, evidence_b = infer_group_id("1_2_000001_0810104210546_NG_part3")
    group_c, _ = infer_group_id("1_2_000001_0810104210546_NG_full")
    assert group_a == group_b == group_c
    assert "part" in evidence_a or "part" in evidence_b


def test_singleton_group_for_classlike_name() -> None:
    group, evidence = infer_group_id("Abnormal thickness156")
    assert group == "Abnormal thickness156"
    assert evidence == "singleton_filename"
