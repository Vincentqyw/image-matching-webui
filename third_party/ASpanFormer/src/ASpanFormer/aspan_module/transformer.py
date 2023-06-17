import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FullAttention, HierachicalAttention ,layernorm2d


class messageLayer_ini(nn.Module):

    def __init__(self, d_model, d_flow,d_value, nhead):
        super().__init__()
        super(messageLayer_ini, self).__init__()

        self.d_model = d_model
        self.d_flow = d_flow
        self.d_value=d_value
        self.nhead = nhead
        self.attention = FullAttention(d_model,nhead)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1,bias=False)
        self.merge_head=nn.Conv1d(d_model,d_model,kernel_size=1,bias=False)

        self.merge_f= self.merge_f = nn.Sequential(
            nn.Conv2d(d_model*2, d_model*2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d_model*2, d_model, kernel_size=1, bias=False),
        )

        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model)


    def forward(self, x0, x1,pos0,pos1,mask0=None,mask1=None):
        #x1,x2: b*d*L
        x0,x1=self.update(x0,x1,pos1,mask0,mask1),\
                self.update(x1,x0,pos0,mask1,mask0)
        return x0,x1


    def update(self,f0,f1,pos1,mask0,mask1):
        """
        Args:
            f0: [N, D, H, W]
            f1: [N, D, H, W]
        Returns:
            f0_new: (N, d, h, w)
        """
        bs,h,w=f0.shape[0],f0.shape[2],f0.shape[3]

        f0_flatten,f1_flatten=f0.view(bs,self.d_model,-1),f1.view(bs,self.d_model,-1)
        pos1_flatten=pos1.view(bs,self.d_value-self.d_model,-1)
        f1_flatten_v=torch.cat([f1_flatten,pos1_flatten],dim=1)

        queries,keys=self.q_proj(f0_flatten),self.k_proj(f1_flatten)
        values=self.v_proj(f1_flatten_v).view(bs,self.nhead,self.d_model//self.nhead,-1)
        
        queried_values=self.attention(queries,keys,values,mask0,mask1)
        msg=self.merge_head(queried_values).view(bs,-1,h,w)
        msg=self.norm2(self.merge_f(torch.cat([f0,self.norm1(msg)],dim=1)))
        return f0+msg



class messageLayer_gla(nn.Module):

    def __init__(self,d_model,d_flow,d_value,
                    nhead,radius_scale,nsample,update_flow=True):
        super().__init__()
        self.d_model = d_model
        self.d_flow=d_flow
        self.d_value=d_value
        self.nhead = nhead
        self.radius_scale=radius_scale
        self.update_flow=update_flow
        self.flow_decoder=nn.Sequential(
                    nn.Conv1d(d_flow, d_flow//2, kernel_size=1, bias=False),
                    nn.ReLU(True),
                    nn.Conv1d(d_flow//2, 4, kernel_size=1, bias=False))
        self.attention=HierachicalAttention(d_model,nhead,nsample,radius_scale)

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1,bias=False)
        self.v_proj = nn.Conv1d(d_value, d_model, kernel_size=1,bias=False)

        d_extra=d_flow if update_flow else 0
        self.merge_f=nn.Sequential(
                     nn.Conv2d(d_model*2+d_extra, d_model+d_flow, kernel_size=1, bias=False),
                     nn.ReLU(True),
                     nn.Conv2d(d_model+d_flow, d_model+d_extra, kernel_size=3,padding=1, bias=False),
                )
        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model+d_extra)

    def forward(self, x0, x1, flow_feature0,flow_feature1,pos0,pos1,mask0=None,mask1=None,ds0=[4,4],ds1=[4,4]):
        """
        Args:
            x0 (torch.Tensor): [B, C, H, W]
            x1 (torch.Tensor): [B, C, H, W]
            flow_feature0 (torch.Tensor): [B, C', H, W]
            flow_feature1 (torch.Tensor): [B, C', H, W]
        """
        flow0,flow1=self.decode_flow(flow_feature0,flow_feature1.shape[2:]),self.decode_flow(flow_feature1,flow_feature0.shape[2:])
        x0_new,flow_feature0_new=self.update(x0,x1,flow0.detach(),flow_feature0,pos1,mask0,mask1,ds0,ds1)
        x1_new,flow_feature1_new=self.update(x1,x0,flow1.detach(),flow_feature1,pos0,mask1,mask0,ds1,ds0)
        return x0_new,x1_new,flow_feature0_new,flow_feature1_new,flow0,flow1

    def update(self,x0,x1,flow0,flow_feature0,pos1,mask0,mask1,ds0,ds1):
        bs=x0.shape[0]
        queries,keys=self.q_proj(x0.view(bs,self.d_model,-1)),self.k_proj(x1.view(bs,self.d_model,-1))
        x1_pos=torch.cat([x1,pos1],dim=1)
        values=self.v_proj(x1_pos.view(bs,self.d_value,-1))
        msg=self.attention(queries,keys,values,flow0,x0.shape[2:],x1.shape[2:],mask0,mask1,ds0,ds1)

        if self.update_flow:
            update_feature=torch.cat([x0,flow_feature0],dim=1)
        else:
            update_feature=x0
        msg=self.norm2(self.merge_f(torch.cat([update_feature,self.norm1(msg)],dim=1)))
        update_feature=update_feature+msg

        x0_new,flow_feature0_new=update_feature[:,:self.d_model],update_feature[:,self.d_model:]
        return x0_new,flow_feature0_new

    def decode_flow(self,flow_feature,kshape):
        bs,h,w=flow_feature.shape[0],flow_feature.shape[2],flow_feature.shape[3]
        scale_factor=torch.tensor([kshape[1],kshape[0]]).cuda()[None,None,None]
        flow=self.flow_decoder(flow_feature.view(bs,-1,h*w)).permute(0,2,1).view(bs,h,w,4)
        flow_coordinates=torch.sigmoid(flow[:,:,:,:2])*scale_factor
        flow_var=flow[:,:,:,2:]
        flow=torch.cat([flow_coordinates,flow_var],dim=-1) #B*H*W*4
        return flow


class flow_initializer(nn.Module):

    def __init__(self, dim, dim_flow, nhead, layer_num):
        super().__init__()
        self.layer_num= layer_num
        self.dim = dim
        self.dim_flow = dim_flow

        encoder_layer = messageLayer_ini(
            dim ,dim_flow,dim+dim_flow , nhead)
        self.layers_coarse = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layer_num)])
        self.decoupler = nn.Conv2d(
                self.dim, self.dim+self.dim_flow, kernel_size=1)
        self.up_merge = nn.Conv2d(2*dim, dim, kernel_size=1)

    def forward(self, feat0, feat1,pos0,pos1,mask0=None,mask1=None,ds0=[4,4],ds1=[4,4]):
        # feat0: [B, C, H0, W0]
        # feat1: [B, C, H1, W1]
        # use low-res MHA to initialize flow feature
        bs = feat0.size(0)
        h0,w0,h1,w1=feat0.shape[2],feat0.shape[3],feat1.shape[2],feat1.shape[3]

        # coarse level
        sub_feat0, sub_feat1 = F.avg_pool2d(feat0, ds0, stride=ds0), \
                            F.avg_pool2d(feat1, ds1, stride=ds1)

        sub_pos0,sub_pos1=F.avg_pool2d(pos0, ds0, stride=ds0), \
                            F.avg_pool2d(pos1, ds1, stride=ds1)
    
        if mask0 is not None:
            mask0,mask1=-F.max_pool2d(-mask0.view(bs,1,h0,w0),ds0,stride=ds0).view(bs,-1),\
                        -F.max_pool2d(-mask1.view(bs,1,h1,w1),ds1,stride=ds1).view(bs,-1)
        
        for layer in self.layers_coarse:
            sub_feat0, sub_feat1 = layer(sub_feat0, sub_feat1,sub_pos0,sub_pos1,mask0,mask1)
        # decouple flow and visual features
        decoupled_feature0, decoupled_feature1 = self.decoupler(sub_feat0),self.decoupler(sub_feat1) 

        sub_feat0, sub_flow_feature0 = decoupled_feature0[:,:self.dim], decoupled_feature0[:, self.dim:]
        sub_feat1, sub_flow_feature1 = decoupled_feature1[:,:self.dim], decoupled_feature1[:, self.dim:]
        update_feat0, flow_feature0 = F.upsample(sub_feat0, scale_factor=ds0, mode='bilinear'),\
                                        F.upsample(sub_flow_feature0, scale_factor=ds0, mode='bilinear')
        update_feat1, flow_feature1 = F.upsample(sub_feat1, scale_factor=ds1, mode='bilinear'),\
                                        F.upsample(sub_flow_feature1, scale_factor=ds1, mode='bilinear')
        
        feat0 = feat0+self.up_merge(torch.cat([feat0, update_feat0], dim=1))
        feat1 = feat1+self.up_merge(torch.cat([feat1, update_feat1], dim=1))
    
        return feat0,feat1,flow_feature0,flow_feature1 #b*c*h*w


class LocalFeatureTransformer_Flow(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer_Flow, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']

        self.pos_transform=nn.Conv2d(config['d_model'],config['d_flow'],kernel_size=1,bias=False)
        self.ini_layer = flow_initializer(self.d_model, config['d_flow'], config['nhead'],config['ini_layer_num'])
        
        encoder_layer = messageLayer_gla(
            config['d_model'], config['d_flow'], config['d_flow']+config['d_model'], config['nhead'],config['radius_scale'],config['nsample'])
        encoder_layer_last=messageLayer_gla(
            config['d_model'], config['d_flow'], config['d_flow']+config['d_model'], config['nhead'],config['radius_scale'],config['nsample'],update_flow=False)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config['layer_num']-1)]+[encoder_layer_last])
        self._reset_parameters()
        
    def _reset_parameters(self):
        for name,p in self.named_parameters():
            if 'temp' in name or 'sample_offset' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1,pos0,pos1,mask0=None,mask1=None,ds0=[4,4],ds1=[4,4]):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            pos1,pos2:  [N, C, H, W]
        Outputs:
            feat0: [N,-1,C]
            feat1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """
        bs = feat0.size(0)
        
        pos0,pos1=self.pos_transform(pos0),self.pos_transform(pos1)
        pos0,pos1=pos0.expand(bs,-1,-1,-1),pos1.expand(bs,-1,-1,-1)
        assert self.d_model == feat0.size(
            1), "the feature number of src and transformer must be equal"
       
        flow_list=[[],[]]# [px,py,sx,sy] 
        if mask0 is not None:
            mask0,mask1=mask0[:,None].float(),mask1[:,None].float()
        feat0,feat1, flow_feature0, flow_feature1 = self.ini_layer(feat0, feat1,pos0,pos1,mask0,mask1,ds0,ds1)
        for layer in self.layers:
            feat0,feat1,flow_feature0,flow_feature1,flow0,flow1=layer(feat0,feat1,flow_feature0,flow_feature1,pos0,pos1,mask0,mask1,ds0,ds1)
            flow_list[0].append(flow0)
            flow_list[1].append(flow1)
        flow_list[0]=torch.stack(flow_list[0],dim=0)
        flow_list[1]=torch.stack(flow_list[1],dim=0)
        feat0, feat1 = feat0.permute(0, 2, 3, 1).view(bs, -1, self.d_model), feat1.permute(0, 2, 3, 1).view(bs, -1, self.d_model)
        return feat0, feat1, flow_list