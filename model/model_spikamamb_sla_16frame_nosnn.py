import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
__all__ = ['spikformer']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_bn = nn.BatchNorm1d(hidden_features)
        # self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        # self.fc2_bn = nn.BatchNorm1d(out_features)
        # self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        # x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        # x = self.fc2_lif(x)
        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)

        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.attn_drop = nn.Dropout(0.5)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N))

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.3, attn_drop=0.,
                 drop_path=0.3, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + (self.mlp(x))
        return x

class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=3, embed_dims=64, name = "1"):
        super().__init__()
        self.name = name
        self.image_size = [img_size_h, img_size_w]
        tuple_patch_size = to_2tuple(patch_size)
        self.num_patch_size = patch_size
        self.patch_size = tuple_patch_size
        self.C = in_channels
        self.embed_dims = embed_dims
        self.H, self.W = self.image_size[0] // tuple_patch_size[0], self.image_size[1] // tuple_patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        # self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        # self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
    def forward(self, x):
        if self.name == "spikmamba":
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = x
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.proj_conv(x) # have some fire value
        x = self.proj_bn(x).reshape(T,B,-1,H,W).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        # x = self.proj_conv3(x)
        # x = self.proj_bn3(x).reshape(T, B, -1, 16, 16).contiguous()
        # x = self.proj_lif3(x).flatten(0, 1).contiguous()
        # x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T,B,-1,H//8,W//8).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)
        x = x + x_rpe
        if self.name == "spikmamba":
            x = x.reshape(T, -1, B, (H//self.num_patch_size), (H//self.num_patch_size)).contiguous()
        else:
            x = x.reshape(B,  self.embed_dims, -1,(H//self.num_patch_size), (H//self.num_patch_size)).contiguous()
        

        return x 
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768, num_frames = 8):
        super().__init__()
        tuple_img_size = to_2tuple(img_size)
        tuple_patch_size = to_2tuple(patch_size)
        num_patches = (tuple_img_size[1] // tuple_patch_size[1]) * (tuple_img_size[0] // tuple_patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size
        self.embed_dim = embed_dim
        self.dropout1 = nn.Dropout3d(0.25)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, tuple_patch_size[0], tuple_patch_size[1]),
            stride=(kernel_size, tuple_patch_size[0], tuple_patch_size[1])
        )
        self.proj_bn = nn.BatchNorm1d(int(1024 ))
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):   
        x = self.proj(x) #
        x = self.dropout1(x)
        B, C, T, H, W = x.shape  
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) 
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.fc2_bn = nn.BatchNorm1d(out_features * 4)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc1_bn(x)
        # x = self.fc1_lif(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.fc2_bn(x)
        # x = self.fc2_lif(x)
        x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0., norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.qk_bn = nn.BatchNorm1d(dim * 4)
        self.qk_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
       
        qk = self.qk(x)
        qk = self.dropout(qk)

        qk = self.qk_lif(self.qk_bn(qk)).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        # qk = self.qk_bn(qk)
        
        # qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x  # q[256,1024,256] k[256,1024,256] v[256,1024,256]
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        a = k.transpose(-2, -1)
        atten = q @ a 
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x, q, k, v, atten

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        # self.dropout1 = nn.Dropout2d(0.25)
        self.in_proj = nn.Linear(dim, dim)
        # self.dropout2 = nn.Dropout(0.25)

        # self.in_bn = nn.BatchNorm1d(dim * 4 )
        # self.in_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.act_proj = nn.Linear(dim, dim)
        # self.act_bn = nn.BatchNorm1d(dim * 4 )
        # self.act_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)
        # self.out_bn = nn.BatchNorm1d(dim * 4 )
        # self.out_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    # def forward(self, x):
    #     H, W = self.input_resolution
    #     B, L, C = x.shape
        
    #     assert L == H * W, "input feature has wrong size"

    #     x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
    #     shortcut = x

    #     x = self.norm1(x)
    #     act_res = self.act(self.act_proj(x))
    #     x = self.in_proj(x).view(B, H, W, C)
    #     x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

    #     # Linear Attention
    #     x = self.attn(x)

    #     x = self.out_proj(x * act_res)
    #     x = shortcut + self.drop_path(x)
    #     x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

    #     # FFN
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x

    def forward(self, x):
        H, W = self.input_resolution
        
        # B, L, C = x.shape
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # x = x.flatten(0, 1)
        a = self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        # a = self.dropout1(a)
        x = x + a
        # x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        x = self.act_proj(x)
        # act_res = self.act(self.act_lif(self.act_bn(x)))
        act_res = self.act(self.act_proj(x))

        # x = self.in_lif(self.in_bn(self.in_proj(x)))
        x = self.in_proj(x).view(B, L, C)
        # x = self.dropout2(x)
        # x = self.in_bn(x)
        # x = self.in_lif(x)
        x = x.view(B, H, W, C)



        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x, q, k, v, atten = self.attn(x)

        # x = self.out_lif(self.out_bn(self.out_proj(x * act_res)))
        x = self.out_proj(x * act_res)
        # x = self.out_bn(x)
        # x = self.out_lif(x)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, q, k, v, atten

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x

class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, q, k, v, atten = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, q, k, v, atten

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=384, img_size_w=384, patch_size=16, in_channels=3, num_classes=300,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0.3, attn_drop_rate=0.3, drop_path_rate=0.3, norm_layer=nn.LayerNorm, num_frames = 8,kernel_size=1,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],batch_size = 32
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.mlp_ratio = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.num_frame = num_frames
        self.sps_patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims,
                                 name="spikeformer")
        
        self.patch_embed = PatchEmbed(
            img_size=img_size_h, patch_size=patch_size, 
            kernel_size=1,
            in_chans=3, embed_dim=embed_dims,
            num_frames = self.num_frame
        )

        self.imgae_size = img_size_h
        self.patch_size = patch_size
        

        num_patches = self.patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        # block = nn.ModuleList([Block(
        #     dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
        #     qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
        #     norm_layer=norm_layer, sr_ratio=sr_ratios)
        #     for j in range(depths)])

        setattr(self, f"patch_embed", self.patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        # setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        self.mlla_norm_layer=nn.LayerNorm
        self.embed_dims =embed_dims
        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dims))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, (num_frames ) // kernel_size, self.embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)
        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)
        patches_resolution = [img_size_h // patch_size, img_size_w // patch_size]
        self.layers = nn.ModuleList()
        self.mlla_depths=[ 2]
        self.mlla_num_heads = [ 8]
        self.num_layers = len(self.mlla_depths)

        self.mlla_img_size = to_2tuple(128)
        self.mlla_patch_size = to_2tuple(4)
        patches_resolution = [self.mlla_img_size[0] // self.mlla_patch_size[0], self.mlla_img_size[1] // self.mlla_patch_size[1]]
        self.mlla_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.mlla_depths))]
        
        
        self.batch_size = batch_size

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.mlla_depths[i_layer],
                               num_heads=self.mlla_num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=self.mlla_dpr[sum(self.mlla_depths[:i_layer]):sum(self.mlla_depths[:i_layer + 1])],
                               norm_layer=self.mlla_norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        # print(self.layers)


    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    def forward_features(self, x):

        # block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # a = x
        x = self.patch_embed(x) #T B L C
        # # for blk in block:
        # #     x = blk(x)
        # x = self.sps_patch_embed(x) # 128 64 8 32 32
        
        

        # B, C, T, H, W = x.shape #   8，256，8，32，32


        # x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) # [128,1024,64]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks [8, 1, 576]
        x = torch.cat((cls_token, x), dim=1)  # [1024, 65, 192]
        x = x + self.pos_embed  # pos_embed [1, 197, 576] x [8, 197, 576]
        cls_tokens = x[:self.batch_size, :1, :] #cls_tokens [1,1,576]
        x = x[:, 1:] # [8, 196, 576]
        x = rearrange(x, '(b t) n m -> (b n) t m', b = int(self.batch_size), t=self.num_frame )
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b = int(self.batch_size) , t=self.num_frame )
        # x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        x = x.reshape(self.imgae_size//self.patch_size, -1 , self.embed_dims).contiguous()
        # 32 8192 256
        x = x.reshape(-1,(self.imgae_size//self.patch_size)*(self.imgae_size//self.patch_size), self.embed_dims).contiguous()
        for layer in self.layers:
            x, q, k, v, atten = layer(x)
        x = x
        x = x.view(int(x.size(0)/self.num_frame),x.size(1),-1,self.embed_dims)
        return x.mean(2),  q, k, v, atten

    def forward(self, x):
        # x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x,  q, k, v, atten = self.forward_features(x)
        x = self.head(x.mean(1))
        return x,  q, k, v, atten



@register_model
def spikmamba_sla_nosnn(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=128, img_size_w=128,patch_size=4, embed_dims=256, num_heads=8 , mlp_ratios=4,
        in_channels=3, num_classes=32, qkv_bias=True,num_frames=16,batch_size = 8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12 , sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 

    return padded_image, padded_mask, meta_mask

def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        # a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    plt.savefig('grid_show+{i}+{j}.png')

def visualize_grid_to_grid_images(att_map, grid_index, image, grid_size=32, alpha=0.6, i=0):
    print("001_0")
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
    
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    # mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = np.mean(att_map, axis=0, keepdims=True).reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig, ax =  plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    
    # ax[0].imshow(grid_image)
    # ax[0].axis('off')
    
    # ax[1].imshow(grid_image)
    # ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='binary_r')
    # ax[1].axis('off')
    # plt.savefig('a.png')
    # plt.show()   binary_r
    ax.imshow(grid_image)
    ax.imshow(mask/np.max(mask), alpha=alpha, cmap='binary_r')
    # ax.axis('off')
    ax.axis('off')

# 保存图片
    save_path = "/home/dsz/Documents/eventcamera/spikformer/cifar10dvs/004/" + str(i) + ".png"
    plt.savefig(save_path)

    plt.show()

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=32, alpha=0.6):
    print("001_0")
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
    
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    # mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = np.mean(att_map, axis=0, keepdims=True).reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig, ax =  plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    
    # ax[0].imshow(grid_image)
    # ax[0].axis('off')
    
    # ax[1].imshow(grid_image)
    # ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='binary_r')
    # ax[1].axis('off')
    # plt.savefig('a.png')
    # plt.show()   binary_r
    ax.imshow(grid_image)
    ax.imshow(mask/np.max(mask), alpha=alpha, cmap='binary_r')
    # ax.axis('off')
    ax.axis('off')

# 保存图片
    save_path = "19_3.png"
    plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import os
    import os
    device = torch.device("cuda:0")

    model = Spikformer(
        img_size_h=128, img_size_w=128,patch_size=4, embed_dims=256, num_heads=8, mlp_ratios=4.0,
        in_channels=3, num_classes=32, qkv_bias=True, num_frames=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1, batch_size = 8
        # **kwargs
    ).cuda()
    

    # model = nn.DataParallel(model).cuda()
    # a = torch.rand(50, 8, 3, 128, 128).cuda(1)
    a = torch.rand(8, 3, 16, 128, 128)
    out = model.forward(a)

    # print(out.shape)

#     data = np.load("/home/dsz/Documents/eventcamera/ExACT/event_npz_16/019/19-s-b-1/19-s-b-1.npz",allow_pickle=True)
#     events = data['images']
#     if events.ndim < 4:
#             events = events[np.newaxis,:,:,:]

#     events_data = np.array(events).transpose(3,0,1,2) / 255.0
#     events_data = torch.from_numpy(events_data)
#     # print(events_data.shape)
#     events_data = events_data.unsqueeze(0).float().to(device)
#     print(events_data.shape)
#     # 加载预训练模型的参数
#     pretrained_dict = torch.load('/home/dsz/Documents/eventcamera/spikformer/cifar10dvs/complete_model_efaction77.17391304347827.pth')
#     torch.save(pretrained_dict.state_dict(), '16frame.pth')
#     pretrained_dict = torch.load('16frame.pth')
#     # 获取新模型的参数字典
#     model_dict = model.state_dict()

#     # 过滤掉不匹配的参数
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

#     # 更新新模型的参数字典
#     model_dict.update(pretrained_dict)

#     # 加载更新后的参数字典到新模型中
#     model.load_state_dict(model_dict)

#     out, q, k, v,atten = model(events_data)
#     # print(out)
#     print(out)
#     print('q',q.size())
#     print('k',k.size())
#     print('v',v.size())
#     print('z',atten.size())


#     # for i in range(17):
#     #     atten_temp = atten[i].unsqueeze(0)
#     #     atten_temp = atten_temp.cpu().detach().numpy()
#     #     picture = str(int(i)) + ".png"
#     #    #  /home/dsz/Documents/eventcamera/ExACT/event_copy_2/026_event/26-s-b-s1
#     #     image_name = "/home/dsz/Documents/eventcamera/ExACT/event copy/019_event/19-s-b-1/" + picture
#     #     image = Image.open(image_name)

#     #     #visualize_grid_to_grid_with_cls(atten[3][0,0,1:,1:], 105, image)

#     #     visualize_grid_to_grid_images(atten_temp[0,7,:,:], 1, image, i = i
#     atten_temp = atten[5].unsqueeze(0)
#     atten_temp = atten_temp.cpu().detach().numpy()
    
#        #  /home/dsz/Documents/eventcamera/ExACT/event_copy_2/026_event/26-s-b-s1
#     image_name = "/home/dsz/Documents/eventcamera/ExACT/19-s-b-1/98.png"
#     image = Image.open(image_name)

# # visualize_grid_to_grid_with_cls(atten[3][0,0,1:,1:], 105, image)

#     visualize_grid_to_grid(atten_temp[0,7,:,:], 1, image)

