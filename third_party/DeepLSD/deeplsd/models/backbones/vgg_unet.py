import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGUNet(torch.nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if tiny:
            sizes = [32, 64, 128, 256]
        else:
            sizes = [64, 128, 256, 512]
        
        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
            nn.Conv2d(sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[3]),
        )

        # Decoder blocks
        self.deblock4 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
        )
        self.deblock3 = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[2]),
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
        )
        self.deblock2 = nn.Sequential(
            nn.Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[1]),
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )
        self.deblock1 = nn.Sequential(
            nn.Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
            nn.Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(sizes[0]),
        )

    def forward(self, inputs):
        # Encoding
        features = [self.block1(inputs)]
        for block in [self.block2, self.block3, self.block4]:
            features.append(block(self.pool(features[-1])))

        # Decoding
        out = self.deblock4(features[-1])
        for deblock, feat in zip(
            [self.deblock3, self.deblock2, self.deblock1], features[:-1][::-1]):
            out = deblock(torch.cat([
                F.interpolate(out, feat.shape[2:4], mode='bilinear'),
                feat], dim=1))

        return out  # dim = 32 if tiny else 64
