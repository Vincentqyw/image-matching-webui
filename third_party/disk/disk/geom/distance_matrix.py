import torch

from torch_dimcheck import dimchecked

SQRT_2 = 1.414213

@dimchecked
def distance_matrix(fs1: ['N', 'F'], fs2: ['M', 'F']) -> ['N', 'M']:
    '''
    Assumes fs1 and fs2 are normalized!
    '''
    return SQRT_2 * (1. - fs1 @ fs2.T).clamp(min=1e-6).sqrt()
