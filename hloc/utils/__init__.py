import os
import logging
logger = logging.getLogger(__name__)
def do_system(cmd,verbose = False):
    if verbose:
        logger.info(f'Run cmd: `{cmd}`.')
    err = os.system(cmd)
    if err:
        logger.info(f'Run cmd err.')
        sys.exit(err)