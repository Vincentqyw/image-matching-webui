import torch
from torch.nn import Module
import torch.nn as nn
from itertools import product
from torch.nn import functional as F

class layernorm2d(nn.Module):
     
     def __init__(self,dim) :
         super().__init__()
         self.dim=dim
         self.affine=nn.parameter.Parameter(torch.ones(dim), requires_grad=True)
         self.bias=nn.parameter.Parameter(torch.zeros(dim), requires_grad=True) 
    
     def forward(self,x):
        #x: B*C*H*W
        mean,std=x.mean(dim=1,keepdim=True),x.std(dim=1,keepdim=True)
        return self.affine[None,:,None,None]*(x-mean)/(std+1e-6)+self.bias[None,:,None,None]


class HierachicalAttention(Module):
    def __init__(self,d_model,nhead,nsample,radius_scale,nlevel=3):
        super().__init__()
        self.d_model=d_model
        self.nhead=nhead
        self.nsample=nsample
        self.nlevel=nlevel
        self.radius_scale=radius_scale
        self.merge_head = nn.Sequential(
            nn.Conv1d(d_model*3, d_model, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, kernel_size=1,bias=False),
        )
        self.fullattention=FullAttention(d_model,nhead)
        self.temp=nn.parameter.Parameter(torch.tensor(1.),requires_grad=True) 
        sample_offset=torch.tensor([[pos[0]-nsample[1]/2+0.5, pos[1]-nsample[1]/2+0.5] for pos in product(range(nsample[1]), range(nsample[1]))]) #r^2*2
        self.sample_offset=nn.parameter.Parameter(sample_offset,requires_grad=False)

    def forward(self,query,key,value,flow,size_q,size_kv,mask0=None, mask1=None,ds0=[4,4],ds1=[4,4]):
        """
        Args:
            q,k,v (torch.Tensor): [B, C, L]
            mask (torch.Tensor): [B, L]
            flow (torch.Tensor): [B, H, W, 4]
        Return:
            all_message (torch.Tensor): [B, C, H, W]
        """
        
        variance=flow[:,:,:,2:]
        offset=flow[:,:,:,:2]  #B*H*W*2
        bs=query.shape[0]
        h0,w0=size_q[0],size_q[1]
        h1,w1=size_kv[0],size_kv[1]
        variance=torch.exp(0.5*variance)*self.radius_scale #b*h*w*2(pixel scale)
        span_scale=torch.clamp((variance*2/self.nsample[1]),min=1) #b*h*w*2

        sub_sample0,sub_sample1=[ds0,2,1],[ds1,2,1]
        q_list=[F.avg_pool2d(query.view(bs,-1,h0,w0),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0]
        k_list=[F.avg_pool2d(key.view(bs,-1,h1,w1),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]
        v_list=[F.avg_pool2d(value.view(bs,-1,h1,w1),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1] #n_level
        
        offset_list=[F.avg_pool2d(offset.permute(0,3,1,2),kernel_size=sub_size*self.nsample[0],stride=sub_size*self.nsample[0]).permute(0,2,3,1)/sub_size for sub_size in sub_sample0[1:]] #n_level-1
        span_list=[F.avg_pool2d(span_scale.permute(0,3,1,2),kernel_size=sub_size*self.nsample[0],stride=sub_size*self.nsample[0]).permute(0,2,3,1) for sub_size in sub_sample0[1:]] #n_level-1

        if mask0 is not None:
            mask0,mask1=mask0.view(bs,1,h0,w0),mask1.view(bs,1,h1,w1)
            mask0_list=[-F.max_pool2d(-mask0,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0]
            mask1_list=[-F.max_pool2d(-mask1,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]
        else:
            mask0_list=mask1_list=[None,None,None]

        message_list=[]
        #full attention at coarse scale
        mask0_flatten=mask0_list[0].view(bs,-1) if mask0 is not None else None
        mask1_flatten=mask1_list[0].view(bs,-1) if mask1 is not None else None
        message_list.append(self.fullattention(q_list[0],k_list[0],v_list[0],mask0_flatten,mask1_flatten,self.temp).view(bs,self.d_model,h0//ds0[0],w0//ds0[1]))

        for index in range(1,self.nlevel):
            q,k,v=q_list[index],k_list[index],v_list[index]
            mask0,mask1=mask0_list[index],mask1_list[index]
            s,o=span_list[index-1],offset_list[index-1] #B*h*w(*2)
            q,k,v,sample_pixel,mask_sample=self.partition_token(q,k,v,o,s,mask0) #B*Head*D*G*N(G*N=H*W for q)
            message_list.append(self.group_attention(q,k,v,1,mask_sample).view(bs,self.d_model,h0//sub_sample0[index],w0//sub_sample0[index]))
        #fuse
        all_message=torch.cat([F.upsample(message_list[idx],scale_factor=sub_sample0[idx],mode='nearest') \
                    for idx in range(self.nlevel)],dim=1).view(bs,-1,h0*w0) #b*3d*H*W
        
        all_message=self.merge_head(all_message).view(bs,-1,h0,w0) #b*d*H*W
        return all_message
      
    def partition_token(self,q,k,v,offset,span_scale,maskv):
        #q,k,v: B*C*H*W
        #o: B*H/2*W/2*2
        #span_scale:B*H*W
        bs=q.shape[0]
        h,w=q.shape[2],q.shape[3]
        hk,wk=k.shape[2],k.shape[3]
        offset=offset.view(bs,-1,2)
        span_scale=span_scale.view(bs,-1,1,2)
        #B*G*2
        offset_sample=self.sample_offset[None,None]*span_scale
        sample_pixel=offset[:,:,None]+offset_sample#B*G*r^2*2
        sample_norm=sample_pixel/torch.tensor([wk/2,hk/2]).cuda()[None,None,None]-1
        
        q = q.view(bs, -1 , h // self.nsample[0], self.nsample[0], w // self.nsample[0], self.nsample[0]).\
                permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, self.nhead,self.d_model//self.nhead, -1,self.nsample[0]**2)#B*head*D*G*N(G*N=H*W for q)
        #sample token
        k=F.grid_sample(k, grid=sample_norm).view(bs, self.nhead,self.d_model//self.nhead,-1, self.nsample[1]**2) #B*head*D*G*r^2
        v=F.grid_sample(v, grid=sample_norm).view(bs, self.nhead,self.d_model//self.nhead,-1, self.nsample[1]**2) #B*head*D*G*r^2
        #import pdb;pdb.set_trace()
        if maskv is not None:
            mask_sample=F.grid_sample(maskv.view(bs,-1,h,w).float(),grid=sample_norm,mode='nearest')==1 #B*1*G*r^2
        else:
            mask_sample=None
        return q,k,v,sample_pixel,mask_sample


    def group_attention(self,query,key,value,temp,mask_sample=None):
        #q,k,v: B*Head*D*G*N(G*N=H*W for q)
        bs=query.shape[0]
        #import pdb;pdb.set_trace()
        QK = torch.einsum("bhdgn,bhdgm->bhgnm", query, key)
        if mask_sample is not None:
            num_head,number_n=QK.shape[1],QK.shape[3]
            QK.masked_fill_(~(mask_sample[:,:,:,None]).expand(-1,num_head,-1,number_n,-1).bool(), float(-1e8))
        # Compute the attention and the weighted average
        softmax_temp = temp / query.size(2)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("bhgnm,bhdgm->bhdgn", A, value).contiguous().view(bs,self.d_model,-1)
        return queried_values

    

class FullAttention(Module):
    def __init__(self,d_model,nhead):
        super().__init__()
        self.d_model=d_model
        self.nhead=nhead

    def forward(self, q, k,v , mask0=None, mask1=None, temp=1):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            q,k,v: [N, D, L]
            mask: [N, L]
        Returns:
            msg: [N,L]
        """
        bs=q.shape[0]
        q,k,v=q.view(bs,self.nhead,self.d_model//self.nhead,-1),k.view(bs,self.nhead,self.d_model//self.nhead,-1),v.view(bs,self.nhead,self.d_model//self.nhead,-1)
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nhdl,nhds->nhls", q, k)
        if mask0 is not None:
            QK.masked_fill_(~(mask0[:,None, :, None] * mask1[:, None, None]).bool(), float(-1e8))
        # Compute the attention and the weighted average
        softmax_temp = temp / q.size(2)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("nhls,nhds->nhdl", A, v).contiguous().view(bs,self.d_model,-1)
        return queried_values
 
    

def elu_feature_map(x):
    return F.elu(x) + 1

class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()