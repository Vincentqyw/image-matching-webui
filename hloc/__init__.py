import logging
import sys

import torch
from packaging import version

__version__ = "1.5"

LOG_PATH = "log.txt"


def read_logs():
    sys.stdout.flush()
    with open(LOG_PATH, "r") as f:
        return f.read()


def flush_logs():
    sys.stdout.flush()
    logs = open(LOG_PATH, "w")
    logs.close()


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

logs_file = open(LOG_PATH, "w")
logs_file.close()

file_handler = logging.FileHandler(filename=LOG_PATH)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger = logging.getLogger("hloc")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
logger.propagate = False

try:
    import pycolmap
except ImportError:
    logger.warning("pycolmap is not installed, some features may not work.")
else:
    min_version = version.parse("0.6.0")
    found_version = pycolmap.__version__
    if found_version != "dev":
        version = version.parse(found_version)
        if version < min_version:
            s = f"pycolmap>={min_version}"
            logger.warning(
                "hloc requires %s but found pycolmap==%s, "
                'please upgrade with `pip install --upgrade "%s"`',
                s,
                found_version,
                s,
            )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model hub: https://huggingface.co/Realcat/imatchui_checkpoint
MODEL_REPO_ID = "Realcat/imatchui_checkpoints"
