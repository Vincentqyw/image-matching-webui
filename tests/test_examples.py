from pathlib import Path

import yaml

from imcui.ui.examples import (
    HARDMATCH_LABELS,
    gen_hardmatch_examples,
    hardmatch_cache_ready,
    hardmatch_required_files,
    load_examples_manifest,
)
from imcui.ui.utils import validate_matcher_zoo

APP_CONFIG = Path(__file__).parents[1] / "imcui" / "config" / "app.yaml"


def test_manifest_valid_and_covers_labels():
    manifest = load_examples_manifest()
    pairs = manifest["hardmatch"]["pairs"]

    assert len(pairs) == 30
    assert {c for pair in pairs for c in pair["categories"]} == HARDMATCH_LABELS
    assert len(hardmatch_required_files(manifest)) == 180

    matchers = manifest["hardmatch"]["settings"]["matchers"]
    zoo = yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8"))["matcher_zoo"]
    for matcher in matchers:
        assert zoo[matcher].get("enable", True)


def test_matcher_labels_follow_convention():
    zoo = yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8"))["matcher_zoo"]
    validate_matcher_zoo(zoo)


def test_rows_shape_offline(tmp_path):
    manifest = load_examples_manifest()
    cache = tmp_path / "hardmatch"
    for rel in hardmatch_required_files(manifest):
        path = cache / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert hardmatch_cache_ready(manifest, cache)
    matchers = manifest["hardmatch"]["settings"]["matchers"]
    rows = gen_hardmatch_examples(manifest, matchers, cache)

    assert len(rows) == 30
    for row in rows:
        assert len(row) == 10
        assert row[5] in matchers
        assert Path(row[0]).is_file()
        assert Path(row[1]).is_file()
    assert {row[5] for row in rows} == set(matchers)
