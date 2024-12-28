import os
import sys
from .. import logger


def do_system(cmd, verbose=False):
    if verbose:
        logger.info(f"Run cmd: `{cmd}`.")
    err = os.system(cmd)
    if err:
        logger.info("Run cmd err.")
        sys.exit(err)
