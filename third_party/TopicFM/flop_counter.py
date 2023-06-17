import torch
from fvcore.nn import FlopCountAnalysis
from einops.einops import rearrange

from src import get_model_cfg
from src.models.backbone import FPN as topicfm_featnet
from src.models.modules import TopicFormer
from src.utils.dataset import read_scannet_gray

from third_party.loftr.src.loftr.utils.cvpr_ds_config import default_cfg
from third_party.loftr.src.loftr.backbone import ResNetFPN_8_2 as loftr_featnet
from third_party.loftr.src.loftr.loftr_module import LocalFeatureTransformer


def feat_net_flops(feat_net, config, input):
    model = feat_net(config)
    model.eval()
    flops = FlopCountAnalysis(model, input)
    feat_c, _ = model(input)
    return feat_c, flops.total() / 1e9


def coarse_model_flops(coarse_model, config, inputs):
    model = coarse_model(config)
    model.eval()
    flops = FlopCountAnalysis(model, inputs)
    return flops.total() / 1e9


if __name__ == '__main__':
    path_img0 = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    path_img1 = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
    img0, img1 = read_scannet_gray(path_img0), read_scannet_gray(path_img1)
    img0, img1 = img0.unsqueeze(0), img1.unsqueeze(0)

    # LoFTR
    loftr_conf = dict(default_cfg)
    feat_c0, loftr_featnet_flops0 = feat_net_flops(loftr_featnet, loftr_conf["resnetfpn"], img0)
    feat_c1, loftr_featnet_flops1 = feat_net_flops(loftr_featnet, loftr_conf["resnetfpn"], img1)
    print("FLOPs of feature extraction in LoFTR: {} GFLOPs".format((loftr_featnet_flops0 + loftr_featnet_flops1)/2))
    feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
    feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
    loftr_coarse_model_flops = coarse_model_flops(LocalFeatureTransformer, loftr_conf["coarse"], (feat_c0, feat_c1))
    print("FLOPs of coarse matching model in LoFTR: {} GFLOPs".format(loftr_coarse_model_flops))

    # TopicFM
    topicfm_conf = get_model_cfg()
    feat_c0, topicfm_featnet_flops0 = feat_net_flops(topicfm_featnet, topicfm_conf["fpn"], img0)
    feat_c1, topicfm_featnet_flops1 = feat_net_flops(topicfm_featnet, topicfm_conf["fpn"], img1)
    print("FLOPs of feature extraction in TopicFM: {} GFLOPs".format((topicfm_featnet_flops0 + topicfm_featnet_flops1) / 2))
    feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
    feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
    topicfm_coarse_model_flops = coarse_model_flops(TopicFormer, topicfm_conf["coarse"], (feat_c0, feat_c1))
    print("FLOPs of coarse matching model in TopicFM: {} GFLOPs".format(topicfm_coarse_model_flops))

