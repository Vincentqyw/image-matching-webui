import os
import logging
import sys
from .. import logger

def do_system(cmd, verbose=False):
    if verbose:
        logger.info(f"Run cmd: `{cmd}`.")
    err = os.system(cmd)
    if err:
        logger.info(f"Run cmd err.")
        sys.exit(err)
