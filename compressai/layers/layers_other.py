# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
import torch
import torch.nn as nn
from .win_attention import WinBasedAttention

__all__ = [
    "conv3x3",
    "subpel_conv3x3",
    "conv1x1",
    "CoCs_BasicLayer",
    "UnshuffleProj",
    "MergingProj",
    "PConv",
    "subpel_PConv3x3",
]

class PConv(nn.Module):
    def __init__(self,dim: int,n_div: int,forward: str = "split_cat",kernel_size: int = 3) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv,self.dim_conv,kernel_size,stride=1,padding=(kernel_size-1)//2,bias=False)

        if forward=="slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat

        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # print(x.shape,self.dim_conv)
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), 1)
        return x
def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
def subpel_PConv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        PConv(in_ch,2),
        conv1x1(in_ch,out_ch * r ** 2),
         nn.PixelShuffle(r)
    )


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self,dim):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.sep = nn.Sequential(nn.Conv2d(dim,dim,3,1,1,groups=dim),
                                 nn.Conv2d(dim,dim,1,1,0))
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1,kernel_size , stride=1, padding=(kernel_size-1)//2),

        )
    def forward(self, x):
        x = self.sep(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()

        self.sep = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                                 nn.Conv2d(channel, channel, 1, 1, 0))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.sep(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)







def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Grouper(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.gap_scaler = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1))
        self.fc1 = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.fc_v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.centers_offset = Mlp_new_offset(head_dim,int(head_dim*1.66), head_dim)
        # self.centers_offset_v = Mlp_new(heads * head_dim, int(heads * head_dim * 1.66), heads * head_dim)

    def forward(self, x):  # [b,c,w,h]
        x = x + self.gap_scaler * (torch.mean(x, dim=[2, 3], keepdim=True))
        # print(x.shape)
        value = self.fc_v(x)
        x = self.fc1(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # splite big feature maps to small patchs to reduce computations of matrix multiplications.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        # print(centers.shape,centers)
        centers = self.centers_offset(x)+centers
        value_centers = rearrange(self.centers_proposal(value)+self.centers_offset(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b, c, -1).permute(0, 2, 1),
                                                              x.reshape(b, c, -1).permute(0, 2, 1)))  # [B,M,N]
        # sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        # return to each point in a cluster
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:  # recover the splited blocks back to big feature maps
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.fc2(out)
        return out


class Mlp_new(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # b,c,w,h ->b,w,h,c
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # b,w,h,c -> b,c,w,h
        return x

class Mlp_new_offset(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.sq = nn.AdaptiveMaxPool2d((2,2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _,_,H,W = x.shape
        x = torch.nn.functional.adaptive_avg_pool2d(x, (H//4,W//4))
        x = x.permute(0, 2, 3, 1)  # b,c,w,h ->b,w,h,c
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # b,w,h,c -> b,c,w,h
        x = self.sq(x)
        return x


class CoCs_BasicLayer(nn.Module):
    def __init__(self, dim, layers, mlp_ratio=2.66, act_layer=nn.GELU, norm_layer=LayerNorm2d, drop_rate=.0,
                 drop_path_rate=0., use_layer_scale=False, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False, pos_k=3):

        super().__init__()
        # build blocks
        blocks = []
        for block_idx in range(layers):
            blocks.append(ClusterBlock(
                dim, mlp_ratio, act_layer, norm_layer, drop_rate, drop_path_rate, use_layer_scale,
                layer_scale_init_value, proposal_w, proposal_h, fold_w, fold_h, heads,
                head_dim, return_center, block_idx
            ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=LayerNorm2d,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False, block_idx=0):

        super().__init__()
        self.use_layer_scale = use_layer_scale

        self.index = block_idx
        if self.index % 2 == 0:
            self.norm1 = norm_layer(dim)


        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
            self.token_mixer = Grouper(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp_new(in_features=dim, hidden_features=mlp_hidden_dim,
                               act_layer=act_layer, drop=drop)

            # The following two techniques are useful to train deep ContextClusters.
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            self.sp = SpatialGate(dim)
            self.ca = SEAttention(dim)
        else:
            self.norm1_an = norm_layer(dim)
            self.norm1_nonan = norm_layer(dim)

            self.token_mixer_an = Grouper(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                       fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim,
                                       return_center=False)
            self.token_mixer_nonan = Grouper(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                          fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim,
                                          return_center=False)
            self.norm2_an = norm_layer(dim)
            self.norm2_nonan = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp_an = Mlp_new(in_features=dim, hidden_features=mlp_hidden_dim,
                               act_layer=act_layer, drop=drop)
            self.mlp_nonan = Mlp_new(in_features=dim, hidden_features=mlp_hidden_dim,
                                  act_layer=act_layer, drop=drop)

            # The following two techniques are useful to train deep ContextClusters.
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            self.sp_an = SpatialGate(dim)
            self.ca_an = SEAttention(dim)
            self.sp_nonan = SpatialGate(dim)
            self.ca_nonan = SEAttention(dim)



    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            if self.index% 2==0:
                # print(self.index)
                x = x + self.drop_path(self.sp(self.token_mixer(self.norm1(x))))
                x = x + self.drop_path(self.ca(self.mlp(self.norm2(x))))
            else:
                # print(self.index)
                anchor = torch.zeros_like(x).to(x.device)
                non_anchor = torch.zeros_like(x).to(x.device)
                anchor[:, :, 0::2, 0::2] = x[:, :, 0::2, 0::2]
                anchor[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]
                non_anchor[:, :, 0::2, 1::2] = x[:, :, 0::2, 1::2]
                non_anchor[:, :, 1::2, 0::2] = x[:, :, 1::2, 0::2]
                anchor = anchor + self.drop_path(self.sp_an(self.token_mixer_an(self.norm1_an(anchor))))
                anchor = anchor + self.drop_path(self.ca_an(self.mlp_an(self.norm2_an(anchor))))
                non_anchor = non_anchor + self.drop_path(self.sp_nonan(self.token_mixer_nonan(self.norm1_nonan(non_anchor))))
                non_anchor = non_anchor + self.drop_path(self.ca_nonan(self.mlp_nonan(self.norm2_nonan(non_anchor))))
                x = anchor+non_anchor


        # x =  x)

        return x


class MergingProj(nn.Module):
    def __init__(self, chans, out_chans,mlp_ratio=2.66):
        super(MergingProj, self).__init__()

        self.mlp = Mlp_new(chans, int(chans * mlp_ratio), chans)
        self.norm = LayerNorm2d(chans * 4)
        self.reduction = nn.Conv2d(4 * chans, out_chans, 1, 1, 0)

    def forward(self, x):
        x = self.mlp(x)

        x0 = x[..., :, 0::2, 0::2]
        x1 = x[..., :, 1::2, 0::2]
        x2 = x[..., :, 0::2, 1::2]
        x3 = x[..., :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class UnshuffleProj(nn.Module):
    def __init__(self, chans, out_chans, r=2):
        super(UnshuffleProj, self).__init__()
        self.UP = nn.Sequential(nn.Conv2d(chans, out_chans * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r))
    def forward(self, x):
        x = self.UP(x)
        return x

