import torch
import torch.nn as nn

from .lcnn_hourglass import MultitaskHead, hg


class HourglassBackbone(nn.Module):
    """ Hourglass backbone. """
    def __init__(self, input_channel=1, depth=4, num_stacks=2,
                 num_blocks=1, num_classes=5):
        super(HourglassBackbone, self).__init__()
        self.head = MultitaskHead
        self.net = hg(**{
            "head": self.head,
            "depth": depth,
            "num_stacks": num_stacks,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "input_channels": input_channel
        })

    def forward(self, input_images):
        return self.net(input_images)[1]


class SuperpointBackbone(nn.Module):
    """ SuperPoint backbone. """
    def __init__(self):
        super(SuperpointBackbone, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4 = 64, 64, 128, 128
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3,
                                      stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3,
                                      stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3,
                                      stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3,
                                      stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3,
                                      stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3,
                                      stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3,
                                      stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3,
                                      stride=1, padding=1)
    
    def forward(self, input_images):
        # Shared Encoder.
        x = self.relu(self.conv1a(input_images))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        return x
