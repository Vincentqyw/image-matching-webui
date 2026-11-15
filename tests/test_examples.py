from pathlib import Path

import pytest
import yaml

from imcui.ui.examples import (
    CATEGORY_LABELS,
    CLASSIC_ROTATIONS,
    CLASSIC_SCALES,
    classic_cache_ready,
    gallery_cache_ready,
    gallery_required_files,
    gen_gallery_examples,
    load_examples_manifest,
)
from imcui.ui.utils import gen_examples, validate_matcher_zoo

APP_CONFIG = Path(__file__).parents[1] / "imcui" / "config" / "app.yaml"


def test_manifest_valid_and_covers_labels():
    manifest = load_examples_manifest()
    assert set(manifest) == {"hardmatch", "fisheye"}

    hardmatch = manifest["hardmatch"]
    assert len(hardmatch["pairs"]) == 30
    assert len(gallery_required_files(manifest, "hardmatch")) == 180

    fisheye = manifest["fisheye"]
    assert len(fisheye["pairs"]) == 6
    assert len(gallery_required_files(manifest, "fisheye")) == 12
    for pair in fisheye["pairs"]:
        assert pair["path"].startswith("fisheye/")
        assert len(pair["source_frames"]) == 2

    categories = {
        category
        for gallery in manifest.values()
        for pair in gallery["pairs"]
        for category in pair["categories"]
    }
    assert categories == CATEGORY_LABELS

    zoo = yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8"))["matcher_zoo"]
    for gallery in manifest.values():
        for matcher in gallery["settings"]["matchers"]:
            assert zoo[matcher].get("enable", True)


def test_matcher_labels_follow_convention():
    zoo = yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8"))["matcher_zoo"]
    validate_matcher_zoo(zoo)


@pytest.mark.parametrize(
    ("name", "expected_rows"),
    [("hardmatch", 30), ("fisheye", 6)],
)
def test_rows_shape_offline(tmp_path, name, expected_rows):
    manifest = load_examples_manifest()
    cache = tmp_path / manifest[name]["cache_subdir"]
    for rel in gallery_required_files(manifest, name):
        path = cache / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert gallery_cache_ready(manifest, name, cache)
    matchers = manifest[name]["settings"]["matchers"]
    rows = gen_gallery_examples(manifest, name, matchers, cache)

    assert len(rows) == expected_rows
    for row in rows:
        assert len(row) == 10
        assert row[5] in matchers
        assert Path(row[0]).is_file()
        assert Path(row[1]).is_file()
    assert {row[5] for row in rows} == set(matchers)


def _write_classic_fixture(root: Path, count: int = 10) -> None:
    mapping = root / "sacre_coeur" / "mapping"
    rotated = root / "sacre_coeur" / "mapping_rot"
    scaled = root / "sacre_coeur" / "mapping_scale"
    for directory in (mapping, rotated, scaled):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        base = f"img{index}"
        (mapping / f"{base}.jpg").write_bytes(b"x")
        for angle in CLASSIC_ROTATIONS:
            (rotated / f"{base}_rot{angle}.jpg").write_bytes(b"x")
        for factor in CLASSIC_SCALES:
            (scaled / f"{base}_scale{factor}.jpg").write_bytes(b"x")


def test_classic_cache_ready(tmp_path):
    root = tmp_path / "imcui_examples"
    _write_classic_fixture(root)
    assert classic_cache_ready(root)

    (root / "sacre_coeur/mapping_rot/img0_rot45.jpg").unlink()
    assert not classic_cache_ready(root)


def test_classic_rows_offline(tmp_path):
    root = tmp_path / "imcui_examples"
    _write_classic_fixture(root)

    rows = gen_examples(root)

    assert len(rows) == 15
    for row in rows:
        assert len(row) == 10
        assert Path(row[0]).is_file()
        assert Path(row[1]).is_file()
