import hashlib
import json
import threading
from collections import OrderedDict

import torch

from ..hloc import logger


class ModelCache:
    """Size-aware LRU cache for models, with a memory budget per device.

    Models are keyed by ``(model_key, model config)`` and moved to the
    highest-priority device that can hold them. Loading a model evicts the
    least recently used models on that device until the model fits; loading a
    model that fits nowhere fails loudly.
    """

    def __init__(
        self,
        max_gpu_mem: float = 8e9,
        max_cpu_mem: float = 12e9,
        device_priority: tuple = ("cuda", "cpu"),
    ):
        """
        Args:
            max_gpu_mem: Maximum GPU memory allowed in bytes.
            max_cpu_mem: Maximum CPU memory allowed in bytes.
            device_priority: Devices to try when loading a model, in order.
        """
        self.max_mem = {"cuda": max_gpu_mem, "cpu": max_cpu_mem}
        self.device_priority = device_priority
        self.cache = OrderedDict()  # key -> {"model", "size", "device"}
        self.current = {"cuda": 0, "cpu": 0}
        self.lock = threading.Lock()
        logger.info("ModelCache initialized.")

    def generate_key(self, model_key, model_conf: dict) -> str:
        # Only the "model" sub-config affects construction. Preprocessing and
        # other per-request values are applied to the cached instance by the
        # caller, so they must not split the cache.
        payload = model_conf.get("model", model_conf)
        unique_str = f"{model_key}-{json.dumps(payload, sort_keys=True)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()

    @staticmethod
    def _model_size(model) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
            b.numel() * b.element_size() for b in model.buffers()
        )

    def _evict(self, device: str, required_size: int) -> bool:
        """Evict LRU models on ``device`` until ``required_size`` bytes are free.

        Returns whether the model fits afterwards.
        """
        for key in [k for k, e in self.cache.items() if e["device"] == device]:
            if self.current[device] + required_size <= self.max_mem[device]:
                break
            entry = self.cache.pop(key)
            self.current[device] -= entry["size"]
            del entry
            if device == "cuda":
                torch.cuda.empty_cache()
        return self.current[device] + required_size <= self.max_mem[device]

    def _select_device(self, size: int) -> str:
        for device in self.device_priority:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            assert device in self.max_mem, f"Unknown device {device}"
            if self._evict(device, size):
                return device
        raise RuntimeError(
            f"Model of {size / 1e9:.2f} GB does not fit in any cached device "
            f"(budgets: {self.max_mem})"
        )

    def load_model(self, model_key, model_loader_func, model_conf: dict):
        key = self.generate_key(model_key, model_conf)

        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]["model"]

            model = model_loader_func(model_conf)
            size = self._model_size(model)
            device = self._select_device(size)
            model = model.to(device)

            self.cache[key] = {"model": model, "size": size, "device": device}
            self.current[device] += size
            return model
