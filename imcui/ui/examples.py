"""Curated example galleries shown in the WebUI.

The manifest in ``imcui/config/examples.yaml`` is the single source of truth for
which HardMatch pairs are shown. Images are downloaded on demand and cached in
the user cache directory; they are never written into the package.
"""

import os
from pathlib import Path

import yaml

EXAMPLES_MANIFEST_PATH = Path(__file__).parents[1] / "config" / "examples.yaml"
EXAMPLES_CACHE_ENV = "IMCUI_EXAMPLES_CACHE"

HARDMATCH_LABELS = {
    "outdoor",
    "building",
    "object",
    "viewpoint",
    "time",
    "illumination",
    "nature",
    "indoor",
    "aerial",
    "celestial",
    "season",
    "dynamic",
    "doppelgänger",
    "drawing",
    "texture",
}

SETTINGS_KEYS = (
    "matcher",
    "match_threshold",
    "max_keypoints",
    "keypoint_threshold",
    "ransac_method",
    "ransac_reproj_threshold",
    "ransac_confidence",
    "ransac_max_iter",
)


def examples_cache_root() -> Path:
    override = os.environ.get(EXAMPLES_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "imcui" / "examples"


def classic_examples_root() -> Path:
    package_data = Path(__file__).parents[1] / "datasets"
    if (package_data / "sacre_coeur" / "mapping").is_dir():
        return package_data
    return examples_cache_root() / "classic"


def load_examples_manifest(path: Path = EXAMPLES_MANIFEST_PATH) -> dict:
    path = Path(path)
    assert path.is_file(), f"Examples manifest not found: {path}"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict) and "hardmatch" in manifest

    hardmatch = manifest["hardmatch"]
    source = hardmatch["source"]
    assert source.get("repo_id"), "Missing HardMatch source repo_id"
    assert source.get("repo_type") == "dataset", "HardMatch source must be a dataset"
    assert source.get("url"), "Missing HardMatch source url"
    assert hardmatch.get("cache_subdir"), "Missing HardMatch cache_subdir"

    files = hardmatch["files"]
    assert files and all(isinstance(name, str) and name for name in files)

    settings = hardmatch["settings"]
    missing = [key for key in SETTINGS_KEYS if key not in settings]
    assert not missing, f"Missing HardMatch settings: {missing}"

    pairs = hardmatch["pairs"]
    assert pairs, "HardMatch manifest has no pairs"
    paths = [pair["path"] for pair in pairs]
    assert len(set(paths)) == len(paths), "Duplicate HardMatch pair paths"
    for pair in pairs:
        pair_path = pair["path"]
        assert pair_path.isascii(), f"Non-ASCII HardMatch path: {pair_path!r}"
        assert (
            pair_path.count("/") == 2
        ), f"Unexpected HardMatch pair path: {pair_path!r}"
        assert pair["title"], f"Missing HardMatch title: {pair_path}"
        categories = pair["categories"]
        assert (
            categories and set(categories) <= HARDMATCH_LABELS
        ), f"Unknown HardMatch categories for {pair_path}: {categories}"
        for side in ("A", "B"):
            attribution = pair["attribution"][side]
            assert attribution["author"], f"Missing author for {pair_path} ({side})"
            assert attribution["license"], f"Missing license for {pair_path} ({side})"
    return manifest


def hardmatch_cache_dir(manifest: dict, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else examples_cache_root()
    return base / manifest["hardmatch"]["cache_subdir"]


def hardmatch_required_files(manifest: dict) -> list[str]:
    hardmatch = manifest["hardmatch"]
    return [
        f"{pair['path']}/{name}"
        for pair in hardmatch["pairs"]
        for name in hardmatch["files"]
    ]


def _is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def hardmatch_cache_ready(manifest: dict, cache_dir: Path | None = None) -> bool:
    cache = Path(cache_dir) if cache_dir is not None else hardmatch_cache_dir(manifest)
    return all(_is_ready(cache / rel) for rel in hardmatch_required_files(manifest))


def download_hardmatch(manifest: dict, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    source = manifest["hardmatch"]["source"]
    cache = Path(cache_dir) if cache_dir is not None else hardmatch_cache_dir(manifest)
    required = hardmatch_required_files(manifest)
    if hardmatch_cache_ready(manifest, cache):
        return cache

    cache.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            revision=source.get("revision", "main"),
            local_dir=str(cache),
            allow_patterns=required,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download HardMatch examples from {source['repo_id']} "
            f"into {cache}. Set {EXAMPLES_CACHE_ENV} to a writable directory "
            f"or check network access to HuggingFace. Original error: {e}"
        ) from e

    missing = [rel for rel in required if not _is_ready(cache / rel)]
    assert not missing, (
        f"HardMatch download finished but {len(missing)} required file(s) are "
        f"missing under {cache}: {missing[:3]}. The remote layout may have "
        "changed. Re-run `imcui --download-examples`."
    )
    return cache


def gen_hardmatch_examples(
    manifest: dict,
    matcher_choices: list[str],
    cache_dir: Path | None = None,
) -> list[list]:
    settings = manifest["hardmatch"]["settings"]
    matcher = settings["matcher"]
    assert (
        matcher in matcher_choices
    ), f"HardMatch example matcher {matcher!r} is not enabled in config/app.yaml"
    cache = Path(cache_dir) if cache_dir is not None else hardmatch_cache_dir(manifest)
    assert hardmatch_cache_ready(manifest, cache), (
        f"HardMatch examples are not cached under {cache}. "
        "Run `imcui --download-examples` first."
    )

    rows = []
    for pair in manifest["hardmatch"]["pairs"]:
        rows.append(
            [
                str(cache / pair["path"] / "A_lr.jpg"),
                str(cache / pair["path"] / "B_lr.jpg"),
                settings["match_threshold"],
                settings["max_keypoints"],
                settings["keypoint_threshold"],
                matcher,
                settings["ransac_method"],
                settings["ransac_reproj_threshold"],
                settings["ransac_confidence"],
                settings["ransac_max_iter"],
            ]
        )
    return rows
