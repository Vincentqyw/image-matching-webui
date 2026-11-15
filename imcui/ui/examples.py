"""Curated example galleries shown in the WebUI.

The manifest in ``imcui/config/examples.yaml`` is the single source of truth for
the fixed-pair galleries. Images are downloaded on demand into the user cache
and are never written into the package.
"""

import os
import random
from pathlib import Path, PurePosixPath

import yaml

EXAMPLES_MANIFEST_PATH = Path(__file__).parents[1] / "config" / "examples.yaml"
EXAMPLES_CACHE_ENV = "IMCUI_EXAMPLES_CACHE"

CLASSIC_REPO_ID = "Realcat/imcui_examples"
CLASSIC_ROTATIONS = (45, 180, 90, 225, 270)
CLASSIC_SCALES = (0.5,)
CLASSIC_PATTERNS = [
    "sacre_coeur/mapping/*.jpg",
    *(f"sacre_coeur/mapping_rot/*_rot{rot}.jpg" for rot in CLASSIC_ROTATIONS),
    *(f"sacre_coeur/mapping_scale/*_scale{scale}.jpg" for scale in CLASSIC_SCALES),
]

CATEGORY_LABELS = {
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
    "matchers",
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
    return examples_cache_root() / "imcui_examples"


def _validate_repo_path(path: str) -> None:
    assert isinstance(path, str) and path, f"Invalid repo path: {path!r}"
    assert path.isascii(), f"Non-ASCII repo path: {path!r}"
    assert "\\" not in path, f"Backslash in repo path: {path!r}"
    pure = PurePosixPath(path)
    assert not pure.is_absolute(), f"Absolute repo path: {path!r}"
    assert all(
        part not in (".", "..") for part in pure.parts
    ), f"Unsafe repo path: {path!r}"
    assert str(pure) == path, f"Unnormalized repo path: {path!r}"


def load_examples_manifest(path: Path = EXAMPLES_MANIFEST_PATH) -> dict:
    path = Path(path)
    assert path.is_file(), f"Examples manifest not found: {path}"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict) and manifest, "Examples manifest is empty"

    cache_subdirs: dict[str, str] = {}
    for name, gallery in manifest.items():
        assert (
            isinstance(name, str) and name.isascii() and name.replace("_", "").isalnum()
        ), f"Invalid gallery name: {name!r}"

        source = gallery["source"]
        assert source["repo_id"], f"Missing {name} source repo_id"
        assert source["repo_type"] == "dataset", f"{name} source must be a dataset"
        assert source["revision"], f"Missing {name} source revision"
        assert source["url"], f"Missing {name} source url"

        cache_subdir = gallery["cache_subdir"]
        assert (
            isinstance(cache_subdir, str)
            and cache_subdir
            and PurePosixPath(cache_subdir).name == cache_subdir
            and cache_subdir not in (".", "..")
        ), f"Invalid {name} cache_subdir: {cache_subdir!r}"
        previous = cache_subdirs.setdefault(source["repo_id"], cache_subdir)
        assert previous == cache_subdir, (
            f"Galleries from {source['repo_id']} must share one cache_subdir; "
            f"got {previous!r} and {cache_subdir!r}"
        )

        ui = gallery["ui"]
        assert ui["title"], f"Missing {name} ui title"
        assert isinstance(ui["open"], bool), f"{name} ui open must be a boolean"
        assert ui["note"], f"Missing {name} ui note"

        files = gallery["files"]
        assert files and all(isinstance(item, str) and item for item in files)
        image_names = gallery["image_names"]
        assert set(image_names) == {"A", "B"}, f"{name} image_names needs A and B"
        assert set(image_names.values()) <= set(
            files
        ), f"{name} image_names must reference files"

        settings = gallery["settings"]
        missing = [key for key in SETTINGS_KEYS if key not in settings]
        assert not missing, f"Missing {name} settings: {missing}"
        matchers = settings["matchers"]
        assert matchers and all(isinstance(item, str) and item for item in matchers)

        pairs = gallery["pairs"]
        assert pairs, f"{name} manifest has no pairs"
        paths = [pair["path"] for pair in pairs]
        assert len(set(paths)) == len(paths), f"Duplicate {name} pair paths"
        for pair in pairs:
            pair_path = pair["path"]
            _validate_repo_path(pair_path)
            assert pair["title"], f"Missing {name} title: {pair_path}"
            categories = pair["categories"]
            assert (
                categories and set(categories) <= CATEGORY_LABELS
            ), f"Unknown {name} categories for {pair_path}: {categories}"
            for side in ("A", "B"):
                attribution = pair["attribution"][side]
                assert attribution["author"], f"Missing author for {pair_path} ({side})"
                assert attribution[
                    "license"
                ], f"Missing license for {pair_path} ({side})"
    return manifest


def gallery_cache_dir(manifest: dict, name: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else examples_cache_root()
    return base / manifest[name]["cache_subdir"]


def gallery_required_files(manifest: dict, name: str) -> list[str]:
    gallery = manifest[name]
    return [
        f"{pair['path']}/{file_name}"
        for pair in gallery["pairs"]
        for file_name in gallery["files"]
    ]


def _is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def gallery_cache_ready(
    manifest: dict, name: str, cache_dir: Path | None = None
) -> bool:
    cache = (
        Path(cache_dir) if cache_dir is not None else gallery_cache_dir(manifest, name)
    )
    return all(_is_ready(cache / rel) for rel in gallery_required_files(manifest, name))


def download_dataset_subset(
    repo_id: str,
    cache_dir: Path,
    patterns: list[str],
    revision: str,
    repo_type: str,
) -> Path:
    from huggingface_hub import snapshot_download

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(cache),
        allow_patterns=patterns,
    )
    return cache


def download_gallery(manifest: dict, name: str, cache_dir: Path | None = None) -> Path:
    gallery = manifest[name]
    source = gallery["source"]
    cache = (
        Path(cache_dir) if cache_dir is not None else gallery_cache_dir(manifest, name)
    )
    required = gallery_required_files(manifest, name)
    if gallery_cache_ready(manifest, name, cache):
        return cache

    try:
        download_dataset_subset(
            source["repo_id"],
            cache,
            required,
            source["revision"],
            source["repo_type"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download {name} examples from {source['repo_id']} "
            f"into {cache}. Set {EXAMPLES_CACHE_ENV} to a writable directory "
            f"or check network access to HuggingFace. Original error: {e}"
        ) from e

    missing = [rel for rel in required if not _is_ready(cache / rel)]
    assert not missing, (
        f"{name} download finished but {len(missing)} required file(s) are "
        f"missing under {cache}: {missing[:3]}. The remote layout may have "
        "changed. Re-run `imcui --download-examples`."
    )
    return cache


def gen_gallery_examples(
    manifest: dict,
    name: str,
    matcher_choices: list[str],
    cache_dir: Path | None = None,
) -> list[list]:
    gallery = manifest[name]
    settings = gallery["settings"]
    matchers = settings["matchers"]
    unknown = [m for m in matchers if m not in matcher_choices]
    assert (
        not unknown
    ), f"{name} example matchers not enabled in config/app.yaml: {unknown}"
    cache = (
        Path(cache_dir) if cache_dir is not None else gallery_cache_dir(manifest, name)
    )
    assert gallery_cache_ready(manifest, name, cache), (
        f"{name} examples are not cached under {cache}. "
        "Run `imcui --download-examples` first."
    )

    pairs = gallery["pairs"]
    image_names = gallery["image_names"]
    # Shuffle matchers in full blocks so every matcher shows up at least once,
    # while the pairing stays fixed across runs (seeded).
    rng = random.Random(0)
    assignment = []
    while len(assignment) < len(pairs):
        block = matchers[:]
        rng.shuffle(block)
        assignment += block

    rows = []
    for pair, matcher in zip(pairs, assignment):
        rows.append(
            [
                str(cache / pair["path"] / image_names["A"]),
                str(cache / pair["path"] / image_names["B"]),
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


def classic_cache_ready(root: Path) -> bool:
    root = Path(root)
    mapping = root / "sacre_coeur" / "mapping"
    if not mapping.is_dir():
        return False
    bases = sorted(
        path.stem
        for path in mapping.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and _is_ready(path)
    )
    if not bases:
        return False
    return all(
        _is_ready(root / "sacre_coeur" / "mapping_rot" / f"{base}_rot{rot}.jpg")
        and _is_ready(
            root / "sacre_coeur" / "mapping_scale" / f"{base}_scale{scale}.jpg"
        )
        for base in bases
        for rot in CLASSIC_ROTATIONS
        for scale in CLASSIC_SCALES
    )


def download_classic_examples(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else classic_examples_root()
    if classic_cache_ready(root):
        return root

    try:
        download_dataset_subset(
            CLASSIC_REPO_ID, root, CLASSIC_PATTERNS, "main", "dataset"
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download classic examples from {CLASSIC_REPO_ID} into "
            f"{root}. Set {EXAMPLES_CACHE_ENV} to a writable directory or check "
            f"network access to HuggingFace. Original error: {e}"
        ) from e

    assert classic_cache_ready(root), (
        f"Downloaded classic examples from {CLASSIC_REPO_ID} into {root}, but "
        "the expected Sacre Coeur variants are missing. The remote layout may "
        "have changed."
    )
    return root
