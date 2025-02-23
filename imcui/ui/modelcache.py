import hashlib
import json
import time
import threading
from collections import OrderedDict
import torch
from ..hloc import logger


class ARCSizeAwareModelCache:
    def __init__(
        self,
        max_gpu_mem: float = 8e9,
        max_cpu_mem: float = 12e9,
        device_priority: list = ["cuda", "cpu"],
        auto_empty_cache: bool = True,
    ):
        """
        Initialize the model cache.

        Args:
            max_gpu_mem: Maximum GPU memory allowed in bytes.
            max_cpu_mem: Maximum CPU memory allowed in bytes.
            device_priority: List of devices to prioritize when evicting models.
            auto_empty_cache: Whether to call torch.cuda.empty_cache() when out of memory.
        """

        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

        self.max_gpu = max_gpu_mem
        self.max_cpu = max_cpu_mem
        self.current_gpu = 0
        self.current_cpu = 0

        self.p = 0
        self.adaptive_factor = 0.5

        self.device_priority = device_priority
        self.lock = threading.Lock()
        self.auto_empty_cache = auto_empty_cache

        logger.info("ARCSizeAwareModelCache initialized.")

    def _release_model(self, model_entry):
        """
        Release a model from memory.

        Args:
            model_entry: A dictionary containing the model, device and other information.

        Notes:
            If the device is CUDA and auto_empty_cache is True, torch.cuda.empty_cache() is called after releasing the model.
        """
        model = model_entry["model"]
        device = model_entry["device"]

        del model
        if device == "cuda":
            torch.cuda.synchronize()
            if self.auto_empty_cache:
                torch.cuda.empty_cache()

    def generate_key(self, model_key, model_conf: dict) -> str:
        loader_identifier = f"{model_key}"
        unique_str = f"{loader_identifier}-{json.dumps(model_conf, sort_keys=True)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def _get_device(self, model_size: int) -> str:
        for device in self.device_priority:
            if device == "cuda" and torch.cuda.is_available():
                if self.current_gpu + model_size <= self.max_gpu:
                    return "cuda"
            elif device == "cpu":
                if self.current_cpu + model_size <= self.max_cpu:
                    return "cpu"
        return "cpu"

    def _calculate_model_size(self, model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
            b.numel() * b.element_size() for b in model.buffers()
        )

    def _update_access(self, key: str, size: int, device: str):
        if key in self.b1:
            self.p = min(
                self.p + max(1, len(self.b2) // len(self.b1)),
                len(self.t1) + len(self.t2),
            )
            self.b1.pop(key)
            self._replace(False)
        elif key in self.b2:
            self.p = max(self.p - max(1, len(self.b1) // len(self.b2)), 0)
            self.b2.pop(key)
            self._replace(True)

        if key in self.t1:
            self.t1.pop(key)
        self.t2[key] = {
            "size": size,
            "device": device,
            "access_count": 1,
            "last_accessed": time.time(),
        }

    def _replace(self, in_t2: bool):
        if len(self.t1) > 0 and (
            (len(self.t1) > self.p) or (in_t2 and len(self.t1) == self.p)
        ):
            k, v = self.t1.popitem(last=False)
            self.b1[k] = v
        else:
            k, v = self.t2.popitem(last=False)
            self.b2[k] = v

    def _calculate_weight(self, entry) -> float:
        return entry["access_count"] / entry["size"]

    def _evict_models(self, required_size: int, target_device: str) -> bool:
        candidates = []
        for k, v in list(self.t1.items()) + list(self.t2.items()):
            if v["device"] == target_device:
                candidates.append((k, v))

        candidates.sort(key=lambda x: self._calculate_weight(x[1]))

        freed = 0
        for k, v in candidates:
            self._release_model(v)
            freed += v["size"]
            if v in self.t1:
                self.t1.pop(k)
            if v in self.t2:
                self.t2.pop(k)

            if v["device"] == "cuda":
                self.current_gpu -= v["size"]
            else:
                self.current_cpu -= v["size"]

            if freed >= required_size:
                return True

        if target_device == "cuda":
            return self._cross_device_evict(required_size, "cuda")
        return False

    def _cross_device_evict(self, required_size: int, target_device: str) -> bool:
        all_entries = []
        for k, v in list(self.t1.items()) + list(self.t2.items()):
            all_entries.append((k, v))

        all_entries.sort(
            key=lambda x: self._calculate_weight(x[1])
            + (0.5 if x[1]["device"] == target_device else 0)
        )

        freed = 0
        for k, v in all_entries:
            freed += v["size"]
            if v in self.t1:
                self.t1.pop(k)
            if v in self.t2:
                self.t2.pop(k)

            if v["device"] == "cuda":
                self.current_gpu -= v["size"]
            else:
                self.current_cpu -= v["size"]

            if freed >= required_size:
                return True
        return False

    def load_model(self, model_key, model_loader_func, model_conf: dict):
        key = self.generate_key(model_key, model_conf)

        with self.lock:
            if key in self.t1 or key in self.t2:
                entry = self.t1.pop(key, None) or self.t2.pop(key)
                entry["access_count"] += 1
                self.t2[key] = entry
                return entry["model"]

            raw_model = model_loader_func(model_conf)
            model_size = self._calculate_model_size(raw_model)
            device = self._get_device(model_size)

            if device == "cuda" and self.auto_empty_cache:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            while True:
                current_mem = self.current_gpu if device == "cuda" else self.current_cpu
                max_mem = self.max_gpu if device == "cuda" else self.max_cpu

                if current_mem + model_size <= max_mem:
                    break

                if not self._evict_models(model_size, device):
                    if device == "cuda":
                        device = "cpu"
                    else:
                        raise RuntimeError("Out of memory")

            try:
                model = raw_model.to(device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    model = raw_model.to(device)

            new_entry = {
                "model": model,
                "size": model_size,
                "device": device,
                "access_count": 1,
                "last_accessed": time.time(),
            }

            if key in self.b1 or key in self.b2:
                self.t2[key] = new_entry
                self._replace(True)
            else:
                self.t1[key] = new_entry
                self._replace(False)

            if device == "cuda":
                self.current_gpu += model_size
            else:
                self.current_cpu += model_size

            return model

    def clear_device_cache(self, device: str):
        with self.lock:
            for cache in [self.t1, self.t2, self.b1, self.b2]:
                for k in list(cache.keys()):
                    if cache[k]["device"] == device:
                        cache.pop(k)


class LRUModelCache:
    def __init__(
        self,
        max_gpu_mem: float = 8e9,
        max_cpu_mem: float = 12e9,
        device_priority: list = ["cuda", "cpu"],
    ):
        self.cache = OrderedDict()
        self.max_gpu = max_gpu_mem
        self.max_cpu = max_cpu_mem
        self.current_gpu = 0
        self.current_cpu = 0
        self.lock = threading.Lock()
        self.device_priority = device_priority

    def generate_key(self, model_key, model_conf: dict) -> str:
        loader_identifier = f"{model_key}"
        unique_str = f"{loader_identifier}-{json.dumps(model_conf, sort_keys=True)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def get_device(self) -> str:
        for device in self.device_priority:
            if device == "cuda" and torch.cuda.is_available():
                if self.current_gpu < self.max_gpu:
                    return device
            elif device == "cpu":
                if self.current_cpu < self.max_cpu:
                    return device
        return "cpu"

    def _calculate_model_size(self, model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    def load_model(self, model_key, model_loader_func, model_conf: dict):
        key = self.generate_key(model_key, model_conf)

        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # update LRU
                return self.cache[key]["model"]

            device = self.get_device()
            if device == "cuda":
                torch.cuda.empty_cache()

            try:
                raw_model = model_loader_func(model_conf)
            except Exception as e:
                raise RuntimeError(f"Model loading failed: {str(e)}")

            try:
                model = raw_model.to(device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    return self._handle_oom(model_key, model_loader_func, model_conf)
                raise

            model_size = self._calculate_model_size(model)

            while (
                device == "cuda" and (self.current_gpu + model_size > self.max_gpu)
            ) or (device == "cpu" and (self.current_cpu + model_size > self.max_cpu)):
                if not self._free_space(model_size, device):
                    raise RuntimeError("Insufficient memory even after cache cleanup")

            if device == "cuda":
                self.current_gpu += model_size
            else:
                self.current_cpu += model_size

            self.cache[key] = {
                "model": model,
                "size": model_size,
                "device": device,
                "timestamp": time.time(),
            }

            return model

    def _free_space(self, required_size: int, device: str) -> bool:
        for key in list(self.cache.keys()):
            if (device == "cuda" and self.cache[key]["device"] == "cuda") or (
                device == "cpu" and self.cache[key]["device"] == "cpu"
            ):
                self.current_gpu -= (
                    self.cache[key]["size"]
                    if self.cache[key]["device"] == "cuda"
                    else 0
                )
                self.current_cpu -= (
                    self.cache[key]["size"] if self.cache[key]["device"] == "cpu" else 0
                )
                del self.cache[key]

                if (
                    device == "cuda"
                    and self.current_gpu + required_size <= self.max_gpu
                ) or (
                    device == "cpu" and self.current_cpu + required_size <= self.max_cpu
                ):
                    return True
        return False

    def _handle_oom(self, model_key, model_loader_func, model_conf: dict):
        with self.lock:
            self.clear_device_cache("cuda")
            torch.cuda.empty_cache()

            try:
                return self.load_model(model_key, model_loader_func, model_conf)
            except RuntimeError:
                original_priority = self.device_priority
                self.device_priority = ["cpu"]
                try:
                    return self.load_model(model_key, model_loader_func, model_conf)
                finally:
                    self.device_priority = original_priority

    def clear_device_cache(self, device: str):
        with self.lock:
            keys_to_remove = [k for k, v in self.cache.items() if v["device"] == device]
            for k in keys_to_remove:
                self.current_gpu -= self.cache[k]["size"] if device == "cuda" else 0
                self.current_cpu -= self.cache[k]["size"] if device == "cpu" else 0
                del self.cache[k]
