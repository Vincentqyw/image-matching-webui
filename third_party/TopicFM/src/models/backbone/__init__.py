from .fpn import FPN


def build_backbone(config):
    return FPN(config['fpn'])
