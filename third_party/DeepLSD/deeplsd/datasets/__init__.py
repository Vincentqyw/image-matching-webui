from ..utils.tools import get_class
from .base_dataset import BaseDataset


def get_dataset(name):
    return get_class(name, __name__, BaseDataset)
