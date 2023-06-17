import torch
import torch.nn.functional as F


def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None
):
    device = feature0.device
    b, c, h, w = feature0.size()
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
                ))
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    r = local_radius
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device=device),
                    torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device=device),
                ))
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(b, 2*r+1, 2*r+1, 2).reshape(b, (2*r+1)**2, 2)
    coords = (coords[:,:,:,None]+local_window[:,None,None]).reshape(b,h,w*(2*r+1)**2,2)
    window_feature = F.grid_sample(
        feature1, coords, padding_mode=padding_mode, align_corners=False
    )[...,None].reshape(b,c,h,w,(2*r+1)**2)
    corr = torch.einsum("bchw, bchwk -> bkhw", feature0, window_feature)/(c**.5)
    return corr
