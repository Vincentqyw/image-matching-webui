import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../../'))
from src.config.default import _CN as cfg

cfg.ASPAN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.ASPAN.MATCH_COARSE.BORDER_RM = 0
cfg.ASPAN.COARSE.COARSEST_LEVEL= [15,20]
cfg.ASPAN.COARSE.TRAIN_RES = [480,640]
