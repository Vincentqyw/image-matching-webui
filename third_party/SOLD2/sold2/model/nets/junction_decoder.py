import torch
import torch.nn as nn


class SuperpointDecoder(nn.Module):
    """ Junction decoder based on the SuperPoint architecture. """
    def __init__(self, input_feat_dim=128, backbone_name="lcnn"):
        super(SuperpointDecoder, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # Perform strided convolution when using lcnn backbone.
        if backbone_name == "lcnn":
            self.convPa = torch.nn.Conv2d(input_feat_dim, 256, kernel_size=3,
                                          stride=2, padding=1)
        elif backbone_name == "superpoint":
            self.convPa = torch.nn.Conv2d(input_feat_dim, 256, kernel_size=3,
                                          stride=1, padding=1)
        else:
            raise ValueError("[Error] Unknown backbone option.")
        
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1,
                                      stride=1, padding=0)

    def forward(self, input_features):
        feat = self.relu(self.convPa(input_features))
        semi = self.convPb(feat)

        return semi