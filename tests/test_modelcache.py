import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from imcui.ui.modelcache import ModelCache

MODEL_SIZE = 24  # nn.Linear(2, 2): 6 parameters * 4 bytes


class CountingLoader:
    def __init__(self):
        self.calls = 0

    def __call__(self, conf):
        self.calls += 1
        return torch.nn.Linear(2, 2)


def _conf(**model):
    return {
        "model": {"name": "fake", **model},
        "preprocessing": {"force_resize": False},
    }


def test_cache_hit_returns_same_model():
    cache = ModelCache(device_priority=("cpu",))
    loader = CountingLoader()
    conf = _conf()

    first = cache.load_model("fake", loader, conf)
    second = cache.load_model("fake", loader, conf)

    assert first is second
    assert loader.calls == 1
    assert cache.current["cpu"] == MODEL_SIZE


def test_key_ignores_preprocessing():
    cache = ModelCache(device_priority=("cpu",))
    a = {"model": {"name": "fake"}, "preprocessing": {"force_resize": True}}
    b = {"model": {"name": "fake"}, "preprocessing": {"force_resize": False}}
    assert cache.generate_key("fake", a) == cache.generate_key("fake", b)


def test_key_tracks_model_config():
    cache = ModelCache(device_priority=("cpu",))
    a = cache.generate_key("fake", _conf(max_keypoints=1))
    b = cache.generate_key("fake", _conf(max_keypoints=2))
    assert a != b


def test_lru_eviction():
    cache = ModelCache(max_cpu_mem=2 * MODEL_SIZE, device_priority=("cpu",))
    loader = CountingLoader()
    confs = [_conf(max_keypoints=i) for i in range(3)]
    keys = [cache.generate_key("fake", conf) for conf in confs]

    cache.load_model("fake", loader, confs[0])
    cache.load_model("fake", loader, confs[1])
    cache.load_model("fake", loader, confs[0])  # hit, confs[1] is now LRU
    cache.load_model("fake", loader, confs[2])  # evicts confs[1]

    assert loader.calls == 3
    assert keys[1] not in cache.cache
    assert keys[0] in cache.cache
    assert keys[2] in cache.cache
    assert cache.current["cpu"] == 2 * MODEL_SIZE


def test_model_too_large_fails_loudly():
    cache = ModelCache(max_cpu_mem=1, device_priority=("cpu",))
    with pytest.raises(RuntimeError, match="does not fit"):
        cache.load_model("fake", CountingLoader(), _conf())
