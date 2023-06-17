from yacs.config import CfgNode
from .config.default import _CN

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CfgNode):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

def get_model_cfg():
    cfg = lower_config(lower_config(_CN))
    return cfg["model"]