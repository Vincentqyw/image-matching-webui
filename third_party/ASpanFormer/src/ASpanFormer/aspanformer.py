import torch
import torch.nn as nn
from torchvision import transforms
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .aspan_module import LocalFeatureTransformer_Flow, LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class ASpanFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],pre_scaling=[config['coarse']['train_res'],config['coarse']['test_res']])
        self.loftr_coarse = LocalFeatureTransformer_Flow(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.coarsest_level=config['coarse']['coarsest_level']

    def forward(self, data, online_resize=False):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        if online_resize:
            assert data['image0'].shape[0]==1 and data['image1'].shape[1]==1
            self.resize_input(data,self.config['coarse']['train_res'])
        else:
            data['pos_scale0'],data['pos_scale1']=None,None

        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(
                torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(
                data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        [feat_c0, pos_encoding0], [feat_c1, pos_encoding1] = self.pos_encoding(feat_c0,data['pos_scale0']), self.pos_encoding(feat_c1,data['pos_scale1'])
        feat_c0 = rearrange(feat_c0, 'n c h w -> n c h w ')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n c h w ')

        #TODO:adjust ds 
        ds0=[int(data['hw0_c'][0]/self.coarsest_level[0]),int(data['hw0_c'][1]/self.coarsest_level[1])]
        ds1=[int(data['hw1_c'][0]/self.coarsest_level[0]),int(data['hw1_c'][1]/self.coarsest_level[1])]
        if online_resize:
            ds0,ds1=[4,4],[4,4]

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(
                -2), data['mask1'].flatten(-2)
        feat_c0, feat_c1, flow_list = self.loftr_coarse(
            feat_c0, feat_c1,pos_encoding0,pos_encoding1,mask_c0,mask_c1,ds0,ds1)

        # 3. match coarse-level and register predicted offset
        self.coarse_matching(feat_c0, feat_c1, flow_list,data,
                             mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # 6. resize match coordinates back to input resolution
        if online_resize:
            data['mkpts0_f']*=data['online_resize_scale0']
            data['mkpts1_f']*=data['online_resize_scale1']
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                if 'sample_offset' in k:
                    state_dict.pop(k)
                else:
                    state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
    def resize_input(self,data,train_res,df=32):
        h0,w0,h1,w1=data['image0'].shape[2],data['image0'].shape[3],data['image1'].shape[2],data['image1'].shape[3]
        data['image0'],data['image1']=self.resize_df(data['image0'],df),self.resize_df(data['image1'],df)
        
        if len(train_res)==1:
            train_res_h=train_res_w=train_res
        else:
            train_res_h,train_res_w=train_res[0],train_res[1]
        data['pos_scale0'],data['pos_scale1']=[train_res_h/data['image0'].shape[2],train_res_w/data['image0'].shape[3]],\
                                  [train_res_h/data['image1'].shape[2],train_res_w/data['image1'].shape[3]] 
        data['online_resize_scale0'],data['online_resize_scale1']=torch.tensor([w0/data['image0'].shape[3],h0/data['image0'].shape[2]])[None].cuda(),\
                                                                    torch.tensor([w1/data['image1'].shape[3],h1/data['image1'].shape[2]])[None].cuda()

    def resize_df(self,image,df=32):
        h,w=image.shape[2],image.shape[3]
        h_new,w_new=h//df*df,w//df*df
        if h!=h_new or w!=w_new:
            img_new=transforms.Resize([h_new,w_new]).forward(image)
        else:
            img_new=image
        return img_new
