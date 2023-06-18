import math
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import get_tuple_transform_ops
from einops import rearrange
from ..utils.local_correlation import local_correlation


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_support_fm = False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_support_fm = no_support_fm
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        device = x.device
        b,c,hs,ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        if self.has_displacement_emb:
            query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-query_coords
            emb_in_displacement = self.disp_emb(in_displacement)
            if self.local_corr_radius:
                #TODO: should corr have gradient?
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow)
                else:
                    # Otherwise we use the warp to sample in the first image
                    # This is actually different operations, especially for large viewpoint changes
                    local_corr = local_correlation(x, x_hat, local_radius=self.local_corr_radius,)
                if self.no_support_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_support_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low (old, new)
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class DFN(nn.Module):
    def __init__(
        self,
        internal_dim,
        feat_input_modules,
        pred_input_modules,
        rrb_d_dict,
        cab_dict,
        rrb_u_dict,
        use_global_context=False,
        global_dim=None,
        terminal_module=None,
        upsample_mode="bilinear",
        align_corners=False,
    ):
        super().__init__()
        if use_global_context:
            assert (
                global_dim is not None
            ), "Global dim must be provided when using global context"
        self.align_corners = align_corners
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.pred_input_modules = pred_input_modules
        self.rrb_d = rrb_d_dict
        self.cab = cab_dict
        self.rrb_u = rrb_u_dict
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_to_internal = nn.Conv2d(global_dim, self.internal_dim, 1, 1, 0)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.terminal_module = (
            terminal_module if terminal_module is not None else nn.Identity()
        )
        self.upsample_mode = upsample_mode
        self._scales = [int(key) for key in self.terminal_module.keys()]

    def scales(self):
        return self._scales.copy()

    def forward(self, embeddings, feats, context, key):
        feats = self.feat_input_modules[str(key)](feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d[str(key)](embeddings)
        context = self.cab[str(key)]([context, embeddings])
        context = self.rrb_u[str(key)](context)
        preds = self.terminal_module[str(key)](context)
        pred_coord = preds[:, -2:]
        pred_certainty = preds[:, :-2]
        return pred_coord, pred_certainty, context


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently supported in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        if self.predict_features:
            f = f + y[:,:self.dim] # Stupid way to predict features
        b, d, h2, w2 = f.shape
        #assert x.shape == y.shape
        x, y, f = self.reshape(x), self.reshape(y), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        # Due to https://github.com/pytorch/pytorch/issues/16963 annoying warnings, remove batch if N large
        if len(K_yy[0]) > 2000:
            K_yy_inv = torch.cat([torch.linalg.inv(K_yy[k:k+1] + sigma_noise[k:k+1]) for k in range(b)])
        else:
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats


class Encoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
    def forward(self, x):
        x0 = x
        b, c, h, w = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x1 = self.resnet.relu(x)

        x = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x)

        x3 = self.resnet.layer2(x2)

        x4 = self.resnet.layer3(x3)

        x5 = self.resnet.layer4(x4)
        feats = {32: x5, 16: x4, 8: x3, 4: x2, 2: x1, 1: x0}
        return feats

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass


class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gps, proj, conv_refiner, transformers = None, detach=False, scales="all", pos_embeddings = None,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales

    def upsample_preds(self, flow, certainty, query, support):
        b, hs, ws, d = flow.shape
        b, c, h, w = query.shape
        flow = flow.permute(0, 3, 1, 2)
        certainty = F.interpolate(
            certainty, size=(h, w), align_corners=False, mode="bilinear"
        )
        flow = F.interpolate(
            flow, size=(h, w), align_corners=False, mode="bilinear"
        )
        delta_certainty, delta_flow = self.conv_refiner["1"](query, support, flow)
        flow = torch.stack(
                (
                    flow[:, 0] + delta_flow[:, 0] / (4 * w),
                    flow[:, 1] + delta_flow[:, 1] / (4 * h),
                ),
                dim=1,
            )
        flow = flow.permute(0, 2, 3, 1)
        certainty = certainty + delta_certainty
        return flow, certainty

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords


    def forward(self, f1, f2, upsample = False, dense_flow = None, dense_certainty = None):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.internal_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        dense_corresps = {}
        if not upsample:
            dense_flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            dense_certainty = 0.0
        else:
            dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        for new_scale in all_scales:
            ins = int(new_scale)
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)
            b, c, hs, ws = f1_s.shape
            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                new_stuff = self.gps[new_scale](f1_s, f2_s, dense_flow=dense_flow)
                dense_flow, dense_certainty, old_stuff = self.embedding_decoder(
                    new_stuff, f1_s, old_stuff, new_scale
                )

            if new_scale in self.conv_refiner:
                delta_certainty, displacement = self.conv_refiner[new_scale](
                    f1_s, f2_s, dense_flow
                )
                dense_flow = torch.stack(
                    (
                        dense_flow[:, 0] + ins * displacement[:, 0] / (4 * w),
                        dense_flow[:, 1] + ins * displacement[:, 1] / (4 * h),
                    ),
                    dim=1,
                )
                dense_certainty = (
                    dense_certainty + delta_certainty
                )  # predict both certainty and displacement

            dense_corresps[ins] = {
                "dense_flow": dense_flow,
                "dense_certainty": dense_certainty,
            }

            if new_scale != "1":
                dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )

                dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )
                if self.detach:
                    dense_flow = dense_flow.detach()
                    dense_certainty = dense_certainty.detach()
        return dense_corresps


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=384,
        w=512,
        use_contrastive_loss = False,
        alpha = 1,
        beta = 0,
        sample_mode = "threshold",
        upsample_preds = False,
        symmetric = False,
        name = None,
        use_soft_mutual_nearest_neighbours = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.use_contrastive_loss = use_contrastive_loss
        self.alpha = alpha
        self.beta = beta
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.symmetric = symmetric
        self.name = name
        self.sample_thresh = 0.05
        self.upsample_res = (864,1152)
        if use_soft_mutual_nearest_neighbours:
            assert symmetric, "MNS requires symmetric inference"
        self.use_soft_mutual_nearest_neighbours = use_soft_mutual_nearest_neighbours
        
    def extract_backbone_features(self, batch, batched = True, upsample = True):
        #TODO: only extract stride [1,2,4,8] for upsample = True
        x_q = batch["query"]
        x_s = batch["support"]
        if batched:
            X = torch.cat((x_q, x_s))
            feature_pyramid = self.encoder(X)
        else:
            feature_pyramid = self.encoder(x_q), self.encoder(x_s)
        return feature_pyramid

    def sample(
        self,
        dense_matches,
        dense_certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        elif "pow" in self.sample_mode:
            dense_certainty = dense_certainty**(1/3)
        elif "naive" in self.sample_mode:
            dense_certainty = torch.ones_like(dense_certainty)
        matches, certainty = (
            dense_matches.reshape(-1, 4),
            dense_certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty

        from ..utils.kde import kde
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, batched = True):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        if self.training and self.use_contrastive_loss:
            return dense_corresps, (f_q_pyramid, f_s_pyramid)
        else:
            return dense_corresps

    def forward_symmetric(self, batch, upsample = False, batched = True):
        feature_pyramid = self.extract_backbone_features(batch, upsample = upsample, batched = batched)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]))
            for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid, upsample = upsample, **(batch["corresps"] if "corresps" in batch else {}))
        return dense_corresps
    
    def to_pixel_coordinates(self, matches, H_A, W_A, H_B, W_B):
        kpts_A, kpts_B = matches[...,:2], matches[...,2:]
        kpts_A = torch.stack((W_A/2 * (kpts_A[...,0]+1), H_A/2 * (kpts_A[...,1]+1)),axis=-1)
        kpts_B = torch.stack((W_B/2 * (kpts_B[...,0]+1), H_B/2 * (kpts_B[...,1]+1)),axis=-1)
        return kpts_A, kpts_B
    
    def match(
        self,
        im1_path,
        im2_path,
        *args,
        batched=False,
        device = None
    ):
        assert not (batched and self.upsample_preds), "Cannot upsample preds if in batchmode (as we don't have access to high res images). You can turn off upsample_preds by model.upsample_preds = False "
        if isinstance(im1_path, (str, os.PathLike)):
            im1, im2 = Image.open(im1_path), Image.open(im2_path)
        else: # assume it is a PIL Image
            im1, im2 = im1_path, im2_path
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im1.size
                w2, h2 = im2.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                query, support = test_transform((im1, im2))
                batch = {"query": query[None].to(device), "support": support[None].to(device)}
            else:
                b, c, h, w = im1.shape
                b, c, h2, w2 = im2.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"query": im1.to(device), "support": im2.to(device)}
                hs, ws = self.h_resized, self.w_resized
            finest_scale = 1
            # Run matcher
            if symmetric:
                dense_corresps  = self.forward_symmetric(batch, batched = True)
            else:
                dense_corresps = self.forward(batch, batched = True)
            
            if self.upsample_preds:
                hs, ws = self.upsample_res
            low_res_certainty = F.interpolate(
            dense_corresps[16]["dense_certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)

            if self.upsample_preds: 
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                query, support = test_transform((im1, im2))
                query, support = query[None].to(device), support[None].to(device)
                batch = {"query": query, "support": support, "corresps": dense_corresps[finest_scale]}
                if symmetric:
                    dense_corresps = self.forward_symmetric(batch, upsample = True, batched=True)
                else:
                    dense_corresps = self.forward(batch, batched = True, upsample=True)
            query_to_support = dense_corresps[finest_scale]["dense_flow"]
            dense_certainty = dense_corresps[finest_scale]["dense_certainty"]
            
            # Get certainty interpolation
            dense_certainty = dense_certainty - low_res_certainty
            query_to_support = query_to_support.permute(
                0, 2, 3, 1
                )
            # Create im1 meshgrid
            query_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            dense_certainty = dense_certainty.sigmoid()  # logits -> probs
            query_coords = query_coords.permute(0, 2, 3, 1)
            if (query_to_support.abs() > 1).any() and True:
                wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
                dense_certainty[wrong[:,None]] = 0
                
            query_to_support = torch.clamp(query_to_support, -1, 1)
            if symmetric:
                support_coords = query_coords
                qts, stq = query_to_support.chunk(2)                    
                q_warp = torch.cat((query_coords, qts), dim=-1)
                s_warp = torch.cat((stq, support_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp),dim=2)
                dense_certainty = torch.cat(dense_certainty.chunk(2), dim=3)[:,0]
            else:
                warp = torch.cat((query_coords, query_to_support), dim=-1)
            if batched:
                return (
                    warp,
                    dense_certainty
                )
            else:
                return (
                    warp[0],
                    dense_certainty[0],
                )
