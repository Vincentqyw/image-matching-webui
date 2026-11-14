from pathlib import Path

import yaml

from imcui.ui.examples import (
    HARDMATCH_LABELS,
    gen_hardmatch_examples,
    hardmatch_cache_ready,
    hardmatch_required_files,
    load_examples_manifest,
)

APP_CONFIG = Path(__file__).parents[1] / "imcui" / "config" / "app.yaml"


def test_manifest_valid_and_covers_labels():
    manifest = load_examples_manifest()
    pairs = manifest["hardmatch"]["pairs"]

    assert len(pairs) == 10
    assert {c for pair in pairs for c in pair["categories"]} == HARDMATCH_LABELS
    assert len(hardmatch_required_files(manifest)) == 60

    matcher = manifest["hardmatch"]["settings"]["matcher"]
    zoo = yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8"))["matcher_zoo"]
    assert zoo[matcher].get("enable", True)


def test_rows_shape_offline(tmp_path):
    manifest = load_examples_manifest()
    cache = tmp_path / "hardmatch"
    for rel in hardmatch_required_files(manifest):
        path = cache / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert hardmatch_cache_ready(manifest, cache)
    matcher = manifest["hardmatch"]["settings"]["matcher"]
    rows = gen_hardmatch_examples(manifest, [matcher], cache)

    assert len(rows) == 10
    for row in rows:
        assert len(row) == 10
        assert row[5] == matcher
        assert Path(row[0]).is_file()
        assert Path(row[1]).is_file()
