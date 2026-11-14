import inspect
import os
import shutil
import sys
from abc import ABCMeta, abstractmethod
from copy import copy
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from .. import MODEL_REPO_ID


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def _download_model(self, repo_id=None, filename=None):
        """Download a model from HuggingFace and return its local path."""
        return hf_hub_download(
            repo_type="model",
            repo_id=repo_id or MODEL_REPO_ID,
            filename=filename,
        )

    def _download_torch_checkpoint(self, filename, cache_name):
        """Seed torch.hub's checkpoint cache from HuggingFace.

        Some third-party models call torch.hub.load_state_dict_from_url with a
        fixed upstream URL. Caching the HuggingFace copy under the same name
        makes that call resolve locally, without patching third_party code.
        """
        dst = Path(torch.hub.get_dir()) / "checkpoints" / cache_name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            shutil.copyfile(self._download_model(filename=filename), tmp)
            tmp.replace(dst)
        return dst


def dynamic_load(root, model):
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)]
    assert len(classes) == 1, classes
    return classes[0][1]
    # return getattr(module, 'Model')
