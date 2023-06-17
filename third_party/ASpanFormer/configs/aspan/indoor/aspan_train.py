import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../../'))
from src.config.default import _CN as cfg

cfg.ASPAN.COARSE.COARSEST_LEVEL= [15,20]
cfg.ASPAN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.ASPAN.MATCH_COARSE.SPARSE_SPVS = False
cfg.ASPAN.MATCH_COARSE.BORDER_RM = 0
cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
