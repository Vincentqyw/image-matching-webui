import torch
import torch.nn as nn


class SuperpointDescriptor(nn.Module):
    """ Descriptor decoder based on the SuperPoint arcihtecture. """
    def __init__(self, input_feat_dim=128):
        super(SuperpointDescriptor, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.convPa = torch.nn.Conv2d(input_feat_dim, 256, kernel_size=3,
                                      stride=1, padding=1)        
        self.convPb = torch.nn.Conv2d(256, 128, kernel_size=1,
                                      stride=1, padding=0)

    def forward(self, input_features):
        feat = self.relu(self.convPa(input_features))
        semi = self.convPb(feat)

        return semi