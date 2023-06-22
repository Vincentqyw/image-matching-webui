import warnings
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.utils.checkpoint
from torch import nn
from .base_model import BaseModel

ETH_EPS = 1e-8


class GlueStick(BaseModel):
    default_conf = {
        'input_dim': 256,
        'descriptor_dim': 256,
        'bottleneck_dim': None,
        'weights': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'num_line_iterations': 1,
        'line_attention': False,
        'filter_threshold': 0.2,
        'checkpointed': False,
        'skip_init': False,
        'inter_supervision': None,
        'loss': {
            'nll_weight': 1.,
            'nll_balancing': 0.5,
            'reward_weight': 0.,
            'bottleneck_l2_weight': 0.,
            'dense_nll_weight': 0.,
            'inter_supervision': [0.3, 0.6],
        },
    }
    required_data_keys = [
        'keypoints0', 'keypoints1',
        'descriptors0', 'descriptors1',
        'keypoint_scores0', 'keypoint_scores1']

    DEFAULT_LOSS_CONF = {'nll_weight': 1., 'nll_balancing': 0.5, 'reward_weight': 0., 'bottleneck_l2_weight': 0.}

    def _init(self, conf):
        if conf.bottleneck_dim is not None:
            self.bottleneck_down = nn.Conv1d(
                conf.input_dim, conf.bottleneck_dim, kernel_size=1)
            self.bottleneck_up = nn.Conv1d(
                conf.bottleneck_dim, conf.input_dim, kernel_size=1)
            nn.init.constant_(self.bottleneck_down.bias, 0.0)
            nn.init.constant_(self.bottleneck_up.bias, 0.0)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Conv1d(
                conf.input_dim, conf.descriptor_dim, kernel_size=1)
            nn.init.constant_(self.input_proj.bias, 0.0)

        self.kenc = KeypointEncoder(conf.descriptor_dim,
                                    conf.keypoint_encoder)
        self.lenc = EndPtEncoder(conf.descriptor_dim, conf.keypoint_encoder)
        self.gnn = AttentionalGNN(conf.descriptor_dim, conf.GNN_layers,
                                  checkpointed=conf.checkpointed,
                                  inter_supervision=conf.inter_supervision,
                                  num_line_iterations=conf.num_line_iterations,
                                  line_attention=conf.line_attention)
        self.final_proj = nn.Conv1d(conf.descriptor_dim, conf.descriptor_dim,
                                    kernel_size=1)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)
        self.final_line_proj = nn.Conv1d(
            conf.descriptor_dim, conf.descriptor_dim, kernel_size=1)
        nn.init.constant_(self.final_line_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_line_proj.weight, gain=1)
        if conf.inter_supervision is not None:
            self.inter_line_proj = nn.ModuleList(
                [nn.Conv1d(conf.descriptor_dim, conf.descriptor_dim, kernel_size=1)
                 for _ in conf.inter_supervision])
            self.layer2idx = {}
            for i, l in enumerate(conf.inter_supervision):
                nn.init.constant_(self.inter_line_proj[i].bias, 0.0)
                nn.init.orthogonal_(self.inter_line_proj[i].weight, gain=1)
                self.layer2idx[l] = i

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        line_bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('line_bin_score', line_bin_score)

        if conf.weights:
            assert isinstance(conf.weights, str)
            state_dict = torch.load(conf.weights, map_location='cpu')
            if 'model' in state_dict:
                state_dict = {k.replace('matcher.', ''): v for k, v in state_dict['model'].items() if 'matcher.' in k}
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict)

    def _forward(self, data):
        device = data['keypoints0'].device
        b_size = len(data['keypoints0'])
        image_size0 = (data['image_size0'] if 'image_size0' in data
                       else data['image0'].shape)
        image_size1 = (data['image_size1'] if 'image_size1' in data
                       else data['image1'].shape)

        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = data['lines0'].shape[1], data['lines1'].shape[1]
        if n_kpts0 == 0 or n_kpts1 == 0:
            # No detected keypoints nor lines
            pred['log_assignment'] = torch.zeros(
                b_size, n_kpts0, n_kpts1, dtype=torch.float, device=device)
            pred['matches0'] = torch.full(
                (b_size, n_kpts0), -1, device=device, dtype=torch.int64)
            pred['matches1'] = torch.full(
                (b_size, n_kpts1), -1, device=device, dtype=torch.int64)
            pred['match_scores0'] = torch.zeros(
                (b_size, n_kpts0), device=device, dtype=torch.float32)
            pred['match_scores1'] = torch.zeros(
                (b_size, n_kpts1), device=device, dtype=torch.float32)
            pred['line_log_assignment'] = torch.zeros(b_size, n_lines0, n_lines1,
                                                      dtype=torch.float, device=device)
            pred['line_matches0'] = torch.full((b_size, n_lines0), -1,
                                               device=device, dtype=torch.int64)
            pred['line_matches1'] = torch.full((b_size, n_lines1), -1,
                                               device=device, dtype=torch.int64)
            pred['line_match_scores0'] = torch.zeros(
                (b_size, n_lines0), device=device, dtype=torch.float32)
            pred['line_match_scores1'] = torch.zeros(
                (b_size, n_kpts1), device=device, dtype=torch.float32)
            return pred

        lines0 = data['lines0'].flatten(1, 2)
        lines1 = data['lines1'].flatten(1, 2)
        lines_junc_idx0 = data['lines_junc_idx0'].flatten(1, 2)  # [b_size, num_lines * 2]
        lines_junc_idx1 = data['lines_junc_idx1'].flatten(1, 2)

        if self.conf.bottleneck_dim is not None:
            pred['down_descriptors0'] = desc0 = self.bottleneck_down(desc0)
            pred['down_descriptors1'] = desc1 = self.bottleneck_down(desc1)
            desc0 = self.bottleneck_up(desc0)
            desc1 = self.bottleneck_up(desc1)
            desc0 = nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = nn.functional.normalize(desc1, p=2, dim=1)
            pred['bottleneck_descriptors0'] = desc0
            pred['bottleneck_descriptors1'] = desc1
            if self.conf.loss.nll_weight == 0:
                desc0 = desc0.detach()
                desc1 = desc1.detach()

        if self.conf.input_dim != self.conf.descriptor_dim:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, image_size0)
        kpts1 = normalize_keypoints(kpts1, image_size1)

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)
        desc0 = desc0 + self.kenc(kpts0, data['keypoint_scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['keypoint_scores1'])

        if n_lines0 != 0 and n_lines1 != 0:
            # Pre-compute the line encodings
            lines0 = normalize_keypoints(lines0, image_size0).reshape(
                b_size, n_lines0, 2, 2)
            lines1 = normalize_keypoints(lines1, image_size1).reshape(
                b_size, n_lines1, 2, 2)
            line_enc0 = self.lenc(lines0, data['line_scores0'])
            line_enc1 = self.lenc(lines1, data['line_scores1'])
        else:
            line_enc0 = torch.zeros(
                b_size, self.conf.descriptor_dim, n_lines0 * 2,
                dtype=torch.float, device=device)
            line_enc1 = torch.zeros(
                b_size, self.conf.descriptor_dim, n_lines1 * 2,
                dtype=torch.float, device=device)

        desc0, desc1 = self.gnn(desc0, desc1, line_enc0, line_enc1,
                                lines_junc_idx0, lines_junc_idx1)

        # Match all points (KP and line junctions)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        kp_scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        kp_scores = kp_scores / self.conf.descriptor_dim ** .5
        kp_scores = log_double_softmax(kp_scores, self.bin_score)
        m0, m1, mscores0, mscores1 = self._get_matches(kp_scores)
        pred['log_assignment'] = kp_scores
        pred['matches0'] = m0
        pred['matches1'] = m1
        pred['match_scores0'] = mscores0
        pred['match_scores1'] = mscores1

        # Match the lines
        if n_lines0 > 0 and n_lines1 > 0:
            (line_scores, m0_lines, m1_lines, mscores0_lines,
             mscores1_lines, raw_line_scores) = self._get_line_matches(
                desc0[:, :, :2 * n_lines0], desc1[:, :, :2 * n_lines1],
                lines_junc_idx0, lines_junc_idx1, self.final_line_proj)
            if self.conf.inter_supervision:
                for l in self.conf.inter_supervision:
                    (line_scores_i, m0_lines_i, m1_lines_i, mscores0_lines_i,
                     mscores1_lines_i) = self._get_line_matches(
                        self.gnn.inter_layers[l][0][:, :, :2 * n_lines0],
                        self.gnn.inter_layers[l][1][:, :, :2 * n_lines1],
                        lines_junc_idx0, lines_junc_idx1,
                        self.inter_line_proj[self.layer2idx[l]])
                    pred[f'line_{l}_log_assignment'] = line_scores_i
                    pred[f'line_{l}_matches0'] = m0_lines_i
                    pred[f'line_{l}_matches1'] = m1_lines_i
                    pred[f'line_{l}_match_scores0'] = mscores0_lines_i
                    pred[f'line_{l}_match_scores1'] = mscores1_lines_i
        else:
            line_scores = torch.zeros(b_size, n_lines0, n_lines1,
                                      dtype=torch.float, device=device)
            m0_lines = torch.full((b_size, n_lines0), -1,
                                  device=device, dtype=torch.int64)
            m1_lines = torch.full((b_size, n_lines1), -1,
                                  device=device, dtype=torch.int64)
            mscores0_lines = torch.zeros(
                (b_size, n_lines0), device=device, dtype=torch.float32)
            mscores1_lines = torch.zeros(
                (b_size, n_lines1), device=device, dtype=torch.float32)
            raw_line_scores = torch.zeros(b_size, n_lines0, n_lines1,
                                          dtype=torch.float, device=device)
        pred['line_log_assignment'] = line_scores
        pred['line_matches0'] = m0_lines
        pred['line_matches1'] = m1_lines
        pred['line_match_scores0'] = mscores0_lines
        pred['line_match_scores1'] = mscores1_lines
        pred['raw_line_scores'] = raw_line_scores

        return pred

    def _get_matches(self, scores_mat):
        max0 = scores_mat[:, :-1, :-1].max(2)
        max1 = scores_mat[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores_mat.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return m0, m1, mscores0, mscores1

    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0,
                          lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)

        line_scores = torch.einsum('bdn,bdm->bnm', mldesc0, mldesc1)
        line_scores = line_scores / self.conf.descriptor_dim ** .5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]
        line_scores = torch.gather(
            line_scores, dim=2,
            index=lines_junc_idx1[:, None, :].repeat(1, line_scores.shape[1], 1))
        line_scores = torch.gather(
            line_scores, dim=1,
            index=lines_junc_idx0[:, :, None].repeat(1, 1, n2_lines1))
        line_scores = line_scores.reshape((-1, n2_lines0 // 2, 2,
                                           n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * torch.maximum(
            line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1],
            line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0])
        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(
            line_scores)
        return (line_scores, m0_lines, m1_lines, mscores0_lines,
                mscores1_lines, raw_line_scores)

    def loss(self, pred, data):
        raise NotImplementedError()

    def metrics(self, pred, data):
        raise NotImplementedError()


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class EndPtEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, endpoints, scores):
        # endpoints should be [B, N, 2, 2]
        # output is [B, feature_dim, N * 2]
        b_size, n_pts, _, _ = endpoints.shape
        assert tuple(endpoints.shape[-2:]) == (2, 2)
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0]).unsqueeze(2)
        endpt_offset = torch.cat([endpt_offset, -endpt_offset], dim=2)
        endpt_offset = endpt_offset.reshape(b_size, 2 * n_pts, 2).transpose(1, 2)
        inputs = [endpoints.flatten(1, 2).transpose(1, 2),
                  endpt_offset, scores.repeat(1, 2).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.prob = []

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob.mean(dim=1))
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        if skip_init:
            self.register_parameter('scaling', nn.Parameter(torch.tensor(0.)))
        else:
            self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type, skip_init):
        super().__init__()
        assert layer_type in ['cross', 'self']
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4, skip_init)

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(nn.Module):
    def __init__(self, feature_dim, line_attention=False):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP([self.dim * 3, self.dim * 2, self.dim], do_bn=True)
        self.line_attention = line_attention
        if line_attention:
            self.proj_node = nn.Conv1d(self.dim, self.dim, kernel_size=1)
            self.proj_neigh = nn.Conv1d(2 * self.dim, self.dim, kernel_size=1)

    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size = lines_junc_idx.shape[0]
        line_desc = torch.gather(
            ldesc, 2, lines_junc_idx[:, None].repeat(1, self.dim, 1))
        message = torch.cat([
            line_desc,
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1)
        return self.mlp(message)  # [b_size, D, n_lines * 2]

    def get_endpoint_attention(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        b_size = lines_junc_idx.shape[0]
        expanded_lines_junc_idx = lines_junc_idx[:, None].repeat(1, self.dim, 1)

        # Query: desc of the current node
        query = self.proj_node(ldesc)  # [b_size, D, n_junc]
        query = torch.gather(query, 2, expanded_lines_junc_idx)
        # query is [b_size, D, n_lines * 2]

        # Key: combination of neighboring desc and line encodings
        line_desc = torch.gather(ldesc, 2, expanded_lines_junc_idx)
        key = self.proj_neigh(torch.cat([
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1))  # [b_size, D, n_lines * 2]

        # Compute the attention weights with a custom softmax per junction
        prob = (query * key).sum(dim=1) / self.dim ** .5  # [b_size, n_lines * 2]
        prob = torch.exp(prob - prob.max())
        denom = torch.zeros_like(ldesc[:, 0]).scatter_reduce_(
            dim=1, index=lines_junc_idx,
            src=prob, reduce='sum', include_self=False)  # [b_size, n_junc]
        denom = torch.gather(denom, 1, lines_junc_idx)  # [b_size, n_lines * 2]
        prob = prob / (denom + ETH_EPS)
        return prob  # [b_size, n_lines * 2]

    def forward(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0,
                lines_junc_idx1):
        # Gather the endpoint updates
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        update0, update1 = torch.zeros_like(ldesc0), torch.zeros_like(ldesc1)
        dim = ldesc0.shape[1]
        if self.line_attention:
            # Compute an attention for each neighbor and do a weighted average
            prob0 = self.get_endpoint_attention(ldesc0, line_enc0,
                                                lines_junc_idx0)
            lupdate0 = lupdate0 * prob0[:, None]
            update0 = update0.scatter_reduce_(
                dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
                src=lupdate0, reduce='sum', include_self=False)
            prob1 = self.get_endpoint_attention(ldesc1, line_enc1,
                                                lines_junc_idx1)
            lupdate1 = lupdate1 * prob1[:, None]
            update1 = update1.scatter_reduce_(
                dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
                src=lupdate1, reduce='sum', include_self=False)
        else:
            # Average the updates for each junction (requires torch > 1.12)
            update0 = update0.scatter_reduce_(
                dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
                src=lupdate0, reduce='mean', include_self=False)
            update1 = update1.scatter_reduce_(
                dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
                src=lupdate1, reduce='mean', include_self=False)

        # Update
        ldesc0 = ldesc0 + update0
        ldesc1 = ldesc1 + update1

        return ldesc0, ldesc1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_types, checkpointed=False,
                 skip=False, inter_supervision=None, num_line_iterations=1,
                 line_attention=False):
        super().__init__()
        self.checkpointed = checkpointed
        self.inter_supervision = inter_supervision
        self.num_line_iterations = num_line_iterations
        self.inter_layers = {}
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type, skip)
            for layer_type in layer_types])
        self.line_layers = nn.ModuleList(
            [LineLayer(feature_dim, line_attention)
             for _ in range(len(layer_types) // 2)])

    def forward(self, desc0, desc1, line_enc0, line_enc1,
                lines_junc_idx0, lines_junc_idx1):
        for i, layer in enumerate(self.layers):
            if self.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    layer, desc0, desc1, preserve_rng_state=False)
            else:
                desc0, desc1 = layer(desc0, desc1)
            if (layer.type == 'self' and lines_junc_idx0.shape[1] > 0
                    and lines_junc_idx1.shape[1] > 0):
                # Add line self attention layers after every self layer
                for _ in range(self.num_line_iterations):
                    if self.checkpointed:
                        desc0, desc1 = torch.utils.checkpoint.checkpoint(
                            self.line_layers[i // 2], desc0, desc1, line_enc0,
                            line_enc1, lines_junc_idx0, lines_junc_idx1,
                            preserve_rng_state=False)
                    else:
                        desc0, desc1 = self.line_layers[i // 2](
                            desc0, desc1, line_enc0, line_enc1,
                            lines_junc_idx0, lines_junc_idx1)

            # Optionally store the line descriptor at intermediate layers
            if (self.inter_supervision is not None
                    and (i // 2) in self.inter_supervision
                    and layer.type == 'cross'):
                self.inter_layers[i // 2] = (desc0.clone(), desc1.clone())
        return desc0, desc1


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
