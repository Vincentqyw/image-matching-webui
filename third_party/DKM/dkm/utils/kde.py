import torch
import torch.nn.functional as F
import numpy as np

def fast_kde(x, std = 0.1, kernel_size = 9, dilation = 3, padding = 9//2, stride = 1):
    raise NotImplementedError("WIP, use at your own risk.")
    # Note: when doing symmetric matching this might not be very exact, since we only check neighbours on the grid
    x = x.permute(0,3,1,2)
    B,C,H,W = x.shape
    K = kernel_size ** 2
    unfolded_x = F.unfold(x,kernel_size=kernel_size, dilation = dilation, padding = padding, stride = stride).reshape(B, C, K, H, W)
    scores = (-(unfolded_x - x[:,:,None]).sum(dim=1)**2/(2*std**2)).exp()
    density = scores.sum(dim=1)
    return density


def kde(x, std = 0.1, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    # use a gaussian kernel to estimate density
    x = x.to(device)
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density
