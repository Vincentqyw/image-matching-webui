weight_urls = {
    "DKMv3": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth",
    },
}
import torch
from .DKMv3 import DKMv3


def DKMv3_outdoor(path_to_weights = None, device=None):
    """
    Loads DKMv3 outdoor weights, uses internal resolution of (540, 720) by default
    resolution can be changed by setting model.h_resized, model.w_resized later.
    Additionally upsamples preds to fixed resolution of (864, 1152),
    can be turned off by model.upsample_preds = False
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path_to_weights is not None:
        weights = torch.load(path_to_weights, map_location='cpu')
    else:
        weights = torch.hub.load_state_dict_from_url(weight_urls["DKMv3"]["outdoor"],
                                                     map_location='cpu')
    return DKMv3(weights, 540, 720, upsample_preds = True, device=device)

def DKMv3_indoor(path_to_weights = None, device=None):
    """
    Loads DKMv3 indoor weights, uses internal resolution of (480, 640) by default
    Resolution can be changed by setting model.h_resized, model.w_resized later.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path_to_weights is not None:
        weights = torch.load(path_to_weights, map_location=device)
    else:
        weights = torch.hub.load_state_dict_from_url(weight_urls["DKMv3"]["indoor"],
                                                     map_location=device)
    return DKMv3(weights, 480, 640, upsample_preds = False, device=device)
