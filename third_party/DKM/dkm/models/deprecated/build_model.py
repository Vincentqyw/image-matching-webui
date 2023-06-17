import torch
import torch.nn as nn
from dkm import *
from .local_corr import LocalCorr
from .corr_channels import NormedCorr
from torchvision.models import resnet as tv_resnet

dkm_pretrained_urls = {
    "DKM": {
        "mega_synthetic": "https://github.com/Parskatt/storage/releases/download/dkm_mega_synthetic/dkm_mega_synthetic.pth",
        "mega": "https://github.com/Parskatt/storage/releases/download/dkm_mega/dkm_mega.pth",
    },
    "DKMv2":{
        "outdoor": "https://github.com/Parskatt/storage/releases/download/dkmv2/dkm_v2_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/dkmv2/dkm_v2_indoor.pth",
    }
}


def DKM(pretrained=True, version="mega_synthetic", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": ConvRefiner(
                2 * 256,
                512,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": ConvRefiner(
                2 * 64,
                128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained),
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            dkm_pretrained_urls["DKM"][version]
        )
        matcher.load_state_dict(weights)
    return matcher

def DKMv2(pretrained=True, version="outdoor", resolution = "low", **kwargs):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128,
                1024+128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
            ),
            "8": ConvRefiner(
                2 * 512+64,
                1024+64,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
            ),
            "4": ConvRefiner(
                2 * 256+32,
                512+32,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
            ),
            "1": ConvRefiner(
                2 * 3+6,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    if resolution == "low":
        h, w = 384, 512
    elif resolution == "high":
        h, w = 480, 640
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained),
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w,**kwargs).to(device)
    if pretrained:
        try:
            weights = torch.hub.load_state_dict_from_url(
                dkm_pretrained_urls["DKMv2"][version]
            )
        except:
            weights = torch.load(
                dkm_pretrained_urls["DKMv2"][version]
            )
        matcher.load_state_dict(weights)
    return matcher


def local_corr(pretrained=True, version="mega_synthetic"):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": LocalCorr(
                81,
                81 * 12,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": LocalCorr(
                81,
                81 * 12,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": LocalCorr(
                81,
                81 * 6,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": LocalCorr(
                81,
                81,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained)
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            dkm_pretrained_urls["local_corr"][version]
        )
        matcher.load_state_dict(weights)
    return matcher


def corr_channels(pretrained=True, version="mega_synthetic"):
    h, w = 384, 512
    gp_dim = (h // 32) * (w // 32), (h // 16) * (w // 16)
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim[0] + feat_dim, dfn_dim),
                "16": RRB(gp_dim[1] + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": ConvRefiner(
                2 * 256,
                512,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": ConvRefiner(
                2 * 64,
                128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    gp32 = NormedCorr()
    gp16 = NormedCorr()
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained)
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            dkm_pretrained_urls["corr_channels"][version]
        )
        matcher.load_state_dict(weights)
    return matcher


def baseline(pretrained=True, version="mega_synthetic"):
    h, w = 384, 512
    gp_dim = (h // 32) * (w // 32), (h // 16) * (w // 16)
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim[0] + feat_dim, dfn_dim),
                "16": RRB(gp_dim[1] + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": LocalCorr(
                81,
                81 * 12,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": LocalCorr(
                81,
                81 * 12,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": LocalCorr(
                81,
                81 * 6,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": LocalCorr(
                81,
                81,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    gp32 = NormedCorr()
    gp16 = NormedCorr()
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained)
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            dkm_pretrained_urls["baseline"][version]
        )
        matcher.load_state_dict(weights)
    return matcher


def linear(pretrained=True, version="mega_synthetic"):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": ConvRefiner(
                2 * 256,
                512,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": ConvRefiner(
                2 * 64,
                128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "linear"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained)
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            dkm_pretrained_urls["linear"][version]
        )
        matcher.load_state_dict(weights)
    return matcher
