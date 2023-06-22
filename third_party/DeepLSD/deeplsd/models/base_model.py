"""
Base class for models.
See mnist_net.py for an example of model.
"""

from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
from torch import nn
from copy import copy


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """
    base_default_conf = {
        'name': None,
        'trainable': True,
    }
    default_conf = {}
    required_data_keys = []
    strict_conf = True

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(
                OmegaConf.create(self.base_default_conf),
                OmegaConf.create(self.default_conf))
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError
