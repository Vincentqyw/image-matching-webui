import torch
import torch.nn as nn
import torch.nn.init as init

from .nets.backbone import HourglassBackbone, SuperpointBackbone
from .nets.junction_decoder import SuperpointDecoder
from .nets.heatmap_decoder import PixelShuffleDecoder
from .nets.descriptor_decoder import SuperpointDescriptor


def get_model(model_cfg=None, loss_weights=None, mode="train"):
    """ Get model based on the model configuration. """
    # Check dataset config is given
    if model_cfg is None:
        raise ValueError("[Error] The model config is required!")

    # List the supported options here
    print("\n\n\t--------Initializing model----------")
    supported_arch = ["simple"]
    if not model_cfg["model_architecture"] in supported_arch:
        raise ValueError(
            "[Error] The model architecture is not in supported arch!")

    if model_cfg["model_architecture"] == "simple":
        model = SOLD2Net(model_cfg)
    else:
        raise ValueError(
            "[Error] The model architecture is not in supported arch!")

    # Optionally register loss weights to the model
    if mode == "train":
        if loss_weights is not None:
            for param_name, param in loss_weights.items():
                if isinstance(param, nn.Parameter):
                    print("\t [Debug] Adding %s with value %f to model"
                          % (param_name, param.item()))
                    model.register_parameter(param_name, param)
        else:
            raise ValueError(
                "[Error] the loss weights can not be None in dynamic weighting mode during training.")

    # Display some summary info.
    print("\tModel architecture: %s" % model_cfg["model_architecture"])
    print("\tBackbone: %s" % model_cfg["backbone"])
    print("\tJunction decoder: %s" % model_cfg["junction_decoder"])
    print("\tHeatmap decoder: %s" % model_cfg["heatmap_decoder"])
    print("\t-------------------------------------")

    return model


class SOLD2Net(nn.Module):
    """ Full network for SOLD². """
    def __init__(self, model_cfg):
        super(SOLD2Net, self).__init__()
        self.name = model_cfg["model_name"]
        self.cfg = model_cfg

        # List supported network options
        self.supported_backbone = ["lcnn", "superpoint"]
        self.backbone_net, self.feat_channel = self.get_backbone()

        # List supported junction decoder options
        self.supported_junction_decoder = ["superpoint_decoder"]
        self.junction_decoder = self.get_junction_decoder()

        # List supported heatmap decoder options
        self.supported_heatmap_decoder = ["pixel_shuffle",
                                          "pixel_shuffle_single"]
        self.heatmap_decoder = self.get_heatmap_decoder()

        # List supported descriptor decoder options
        if "descriptor_decoder" in self.cfg:
            self.supported_descriptor_decoder = ["superpoint_descriptor"]
            self.descriptor_decoder = self.get_descriptor_decoder()

        # Initialize the model weights
        self.apply(weight_init)

    def forward(self, input_images):
        # The backbone
        features = self.backbone_net(input_images)

        # junction decoder
        junctions = self.junction_decoder(features)

        # heatmap decoder
        heatmaps = self.heatmap_decoder(features)

        outputs = {"junctions": junctions, "heatmap": heatmaps}

        # Descriptor decoder
        if "descriptor_decoder" in self.cfg:
            outputs["descriptors"] = self.descriptor_decoder(features)

        return outputs

    def get_backbone(self):
        """ Retrieve the backbone encoder network. """
        if not self.cfg["backbone"] in self.supported_backbone:
            raise ValueError(
                "[Error] The backbone selection is not supported.")

        # lcnn backbone (stacked hourglass)
        if self.cfg["backbone"] == "lcnn":
            backbone_cfg = self.cfg["backbone_cfg"]
            backbone = HourglassBackbone(**backbone_cfg)
            feat_channel = 256

        elif self.cfg["backbone"] == "superpoint":
            backbone_cfg = self.cfg["backbone_cfg"]
            backbone = SuperpointBackbone()
            feat_channel = 128

        else:
            raise ValueError(
                "[Error] The backbone selection is not supported.")

        return backbone, feat_channel

    def get_junction_decoder(self):
        """ Get the junction decoder. """
        if (not self.cfg["junction_decoder"]
            in self.supported_junction_decoder):
            raise ValueError(
                "[Error] The junction decoder selection is not supported.")

        # superpoint decoder
        if self.cfg["junction_decoder"] == "superpoint_decoder":
            decoder = SuperpointDecoder(self.feat_channel,
                                        self.cfg["backbone"])
        else:
            raise ValueError(
                "[Error] The junction decoder selection is not supported.")

        return decoder

    def get_heatmap_decoder(self):
        """ Get the heatmap decoder. """
        if not self.cfg["heatmap_decoder"] in self.supported_heatmap_decoder:
            raise ValueError(
                "[Error] The heatmap decoder selection is not supported.")

        # Pixel_shuffle decoder
        if self.cfg["heatmap_decoder"] == "pixel_shuffle":
            if self.cfg["backbone"] == "lcnn":
                decoder = PixelShuffleDecoder(self.feat_channel,
                                              num_upsample=2)
            elif self.cfg["backbone"] == "superpoint":
                decoder = PixelShuffleDecoder(self.feat_channel,
                                              num_upsample=3)
            else:
                raise ValueError("[Error] Unknown backbone option.")
        # Pixel_shuffle decoder with single channel output
        elif self.cfg["heatmap_decoder"] == "pixel_shuffle_single":
            if self.cfg["backbone"] == "lcnn":
                decoder = PixelShuffleDecoder(
                    self.feat_channel, num_upsample=2, output_channel=1)
            elif self.cfg["backbone"] == "superpoint":
                decoder = PixelShuffleDecoder(
                    self.feat_channel, num_upsample=3, output_channel=1)
            else:
                raise ValueError("[Error] Unknown backbone option.")
        else:
            raise ValueError(
                "[Error] The heatmap decoder selection is not supported.")

        return decoder

    def get_descriptor_decoder(self):
        """ Get the descriptor decoder. """
        if (not self.cfg["descriptor_decoder"]
            in self.supported_descriptor_decoder):
            raise ValueError(
                "[Error] The descriptor decoder selection is not supported.")

        # SuperPoint descriptor
        if self.cfg["descriptor_decoder"] == "superpoint_descriptor":
            decoder = SuperpointDescriptor(self.feat_channel)
        else:
            raise ValueError(
                "[Error] The descriptor decoder selection is not supported.")

        return decoder


def weight_init(m):
    """ Weight initialization function. """
    # Conv2D
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    # Batchnorm
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    # Linear
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    else:
        pass
