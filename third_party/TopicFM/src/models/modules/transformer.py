from loguru import logger
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.GELU(),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class TopicFormer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(TopicFormer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        self.topic_transformers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2*config['n_topic_transformers'])]) if config['n_samples'] > 0 else None #nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2)])
        self.n_iter_topic_transformer = config['n_topic_transformers']

        self.seed_tokens = nn.Parameter(torch.randn(config['n_topics'], config['d_model']))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.n_samples = config['n_samples']

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        """
        Args:
            topics (torch.Tensor): [N, L+S, K]
        """
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1  = topics[:, :L], topics[:, L:]

        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1) # [N, K]
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
            sampled_values = torch.gather(theta, dim=-1, index=sampled_inds)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float32, device=feat.device)
        new_mask = torch.zeros_like(resized_feat[..., 0]).bool()
        for i in range(N):
            new_mask[i, :len_topic[i]] = True
        resized_feat[new_mask, :] = feat[selected_ids, :]
        return resized_feat, new_mask, selected_ids

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"
        N, L, S, C, K = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat0.shape[2], self.config['n_topics']

        seeds = self.seed_tokens.unsqueeze(0).repeat(N, 1, 1)

        feat = torch.cat((feat0, feat1), dim=1)
        if mask0 is not None:
            mask = torch.cat((mask0, mask1), dim=-1)
        else:
            mask = None

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'seed':
                # seeds = layer(seeds, feat0, None, mask0)
                # seeds = layer(seeds, feat1, None, mask1)
                seeds = layer(seeds, feat, None, mask)
            elif name == 'feat':
                feat0 = layer(feat0, seeds, mask0, None)
                feat1 = layer(feat1, seeds, mask1, None)

        dmatrix = torch.einsum("nmd,nkd->nmk", feat, seeds)
        prob_topics = F.softmax(dmatrix, dim=-1)

        feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)

        if mask is not None:
            feat_topics = feat_topics * mask.unsqueeze(-1)
            prob_topics = prob_topics * mask.unsqueeze(-1)

        if (feat_topics.detach().sum(dim=1).sum(dim=0) > 100).sum() <= 3:
            logger.warning("topic distribution is highly sparse!")
        sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)
        if sampled_topics is not None:
            updated_feat0, updated_feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
            s_topics0, s_topics1 = sampled_topics
            for k in range(s_topics0.shape[-1]):
                topick0, topick1 = s_topics0[..., k], s_topics1[..., k] # [N, L+S]
                if (topick0.sum() > 0) and (topick1.sum() > 0):
                    new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feat0, topick0, N, C)
                    new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feat1, topick1, N, C)
                    for idt in range(self.n_iter_topic_transformer):
                        new_feat0 = self.topic_transformers[idt*2](new_feat0, new_feat0, new_mask0, new_mask0)
                        new_feat1 = self.topic_transformers[idt*2](new_feat1, new_feat1, new_mask1, new_mask1)
                        new_feat0 = self.topic_transformers[idt*2+1](new_feat0, new_feat1, new_mask0, new_mask1)
                        new_feat1 = self.topic_transformers[idt*2+1](new_feat1, new_feat0, new_mask1, new_mask0)
                    updated_feat0[selected_ids0, :] = new_feat0[new_mask0, :]
                    updated_feat1[selected_ids1, :] = new_feat1[new_mask1, :]

            feat0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feat0 + updated_feat0
            feat1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feat1 + updated_feat1

        conf_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / C**.5 #(C * temperature)
        if self.training:
            topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
            outlier_mask = torch.einsum("nlk,nsk->nls", feat_topics[:, :L], feat_topics[:, L:])
        else:
            topic_matrix = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}
            outlier_mask = torch.ones_like(conf_matrix)
        if mask0 is not None:
            outlier_mask = (outlier_mask * mask0[..., None] * mask1[:, None]) #.bool()
        conf_matrix.masked_fill_(~outlier_mask.bool(), -1e9)
        conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)  # * topic_matrix

        return feat0, feat1, conf_matrix, topic_matrix


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2)]) #len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"

        feat0 = self.layers[0](feat0, feat1, mask0, mask1)
        feat1 = self.layers[1](feat1, feat0, mask1, mask0)

        return feat0, feat1
