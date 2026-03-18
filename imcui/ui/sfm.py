# sfm.py - Disabled for vismatch refactoring
# This module was using hloc which has been removed.
# SfM functionality will be re-enabled in a future update.

import logging

logger = logging.getLogger("imcui.sfm")
logger.warning("SfM module is temporarily disabled due to vismatch refactoring")


class SfmEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg
        logger.warning("SfM is not yet compatible with vismatch")

    def call(self, *args, **kwargs):
        logger.warning("SfM is not yet compatible with vismatch")
        return None, None
