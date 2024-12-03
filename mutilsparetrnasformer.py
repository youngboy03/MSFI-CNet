# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class ASPPCV(nn.Sequential):
    """
    ASPP卷积模块的定义
    """

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPCV, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    ASPP的pooling层
    """

    def __init__(self, in_channels, out_channels):  # [in_channel=out_channel=256]
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # [256*1*1]
            # 自适应平均池化层，只需要给定输出的特征图的尺寸(括号内数字)就好了
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP空洞卷积块
    """

    def __init__(self, in_channels, atrous_rates=(3,5)):  # atrous_rates=(6, 12, 18)
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        rate1, rate2 = tuple(atrous_rates)
        modules.append(ASPPCV(in_channels, out_channels, rate1))  # 3*3卷积( padding=6, dilation=6 )
        modules.append(ASPPCV(in_channels, out_channels, rate2))  # 3*3 卷积( padding=12, dilation=12 )


        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(  # 特征融合？此时输入通道是原始输入通道的5倍。输出的结果又回到原始的通道数。
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),  # [1280*64*64]
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        out = self.project(res)  # 特征融合1280——>256
        # return x + net
        return out

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride
                              )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, patch_size,dim,num_heads,stride, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q=ASPP(dim)


        self.PE=OverlapPatchEmbed(patch_size=patch_size,stride=stride,in_chans=dim,embed_dim=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x=x+self.PE(x)
        q=self.q(x)

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)#8.64.64.64

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)#8.2.32.4096
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N= q.shape

        mask1 = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)#8.2.32.32
        mask2 = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature#8.2.32.32

        index = torch.topk(attn, k=int(N/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(N*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(N*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(N*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v.transpose(-2, -1))
        out2 = (attn2 @ v.transpose(-2, -1))
        out3 = (attn3 @ v.transpose(-2, -1))
        out4 = (attn4 @ v.transpose(-2, -1))

        out = self.attn1 * out1 + self.attn2 * out2 + self.attn3 * out3 + self.attn4 * out4

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x
# class Downsample(nn.Module):Channel downsampling
#     def __init__(self, n_feat):
#         super(Downsample, self).__init__()
#
#         self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
#                                   nn.PixelUnshuffle(2))
#
#     def forward(self, x):
#         return self.body(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.body(x)
class TransformerBlock(nn.Module):
    def __init__(self, patch_size,dim, num_heads, ffn_expansion_factor,stride, bias=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = Attention(patch_size=patch_size,dim=dim, num_heads=num_heads,stride=stride,bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.down1_2 = Downsample(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.attn(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        #x = x + self.ffn(x)
        x=self.down1_2(x)


        return x

if __name__ == '__main__':
    a = torch.randn(8,64,17,17)
    device = torch.device('cpu')

    model = TransformerBlock(patch_size=1,dim=64,num_heads=2,ffn_expansion_factor=4,stride=1).to(device)
    o = model(a)



    print(o.shape)