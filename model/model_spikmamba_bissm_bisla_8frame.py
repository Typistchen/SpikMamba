import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
import torch.utils.checkpoint as checkpoint
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from einops import rearrange
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

class SSM_Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            ) # hidden_states [1, 1569, 576]
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

#     def forward(self, x):
#         x = x + self.attn(x)
#         x = x + (self.mlp(x))
#         return x

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
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, tuple_patch_size[0], tuple_patch_size[1]),
            stride=(kernel_size, tuple_patch_size[0], tuple_patch_size[1])
        )
        self.proj_bn = nn.BatchNorm1d(int(256 ))
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):   
        x = self.proj(x) #
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
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
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

        self.qk_bn = nn.BatchNorm1d(dim )
        self.qk_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

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
       
        qk = self.qk_lif(self.qk_bn(self.qk(x) )).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        # qk = self.qk_bn(qk)
        
        # qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

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

        self.in_proj = nn.Linear(dim, dim)
        # self.in_bn = nn.BatchNorm1d(dim * 4 )
        # self.in_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.to2 = nn.Linear(dim, dim * 2)
        self.act_proj = nn.Linear(dim, dim)
        # self.act_bn = nn.BatchNorm1d(dim * 4 )
        # self.act_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)

        self.dim2 = nn.Linear(512, 256)
        self.out_proj = nn.Linear(dim, dim)
        # self.out_bn = nn.BatchNorm1d(dim * 4 )
        # self.out_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        
        # B, L, C = x.shape
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # x = x.flatten(0, 1)
        
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        # x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        # x2 = self.to2(x)
        # xf, xb = torch.chunk(x2, 2, dim=2)
        xb = x.flip(1)
        xz = torch.cat([x, xb], dim=1)


        xz = self.act_proj(xz)
        # act_res = self.act(self.act_lif(self.act_bn(x)))
        act_res = self.act(self.act_proj(xz))

        # x = self.in_lif(self.in_bn(self.in_proj(x)))
        xz = self.in_proj(xz).view(B, L*2, C)
        # x = self.in_bn(x)
        # x = self.in_lif(x)
        xz = xz.view(B, H*2, W, C)



        xz = self.act(self.dwc(xz.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L*2, C)
        xf,xb = xz.chunk(2,dim=1)
        # Linear Attention
        xf = self.attn(xf)
        xb = self.attn(xb)
        # 
        xz = torch.cat([xf, xb.flip(1)], dim=1)

        # x = self.out_lif(self.out_bn(self.out_proj(x * act_res)))
        
        xz = self.out_proj(xz * act_res)
        # x = self.out_bn(x)
        # x = self.out_lif(x)
        xz = self.dim2(xz.permute(0,2,1)).permute(0, 2,1)
        xz = shortcut + self.drop_path(xz)
        xz = x + self.cpe2(xz.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        xz = xz + self.drop_path(self.mlp(self.norm2(xz)))
        
        return xz

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
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

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
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames = 8,kernel_size=1,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],batch_size = 32,
                 ssm_embed_dims = 256,
                 ssm_cfg=None,
                 norm_epsilon=1e-5, 
                 rms_norm=True, 
                 residual_in_fp32=True,
                 fused_add_norm=True,
                 bimamba=True,
                 device=None,
                 dtype=None
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
        self.mlla_patch_size = to_2tuple(8)
        patches_resolution = [self.mlla_img_size[0] // self.mlla_patch_size[0], self.mlla_img_size[1] // self.mlla_patch_size[1]]
        self.mlla_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.mlla_depths))]

        # smm
        ssm_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]  # stochastic depth decay rule
        ssm_inter_dpr = [0.0] + ssm_dpr
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(ssm_embed_dims, eps=norm_epsilon, **factory_kwargs)
        # self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(ssm_embed_dims, eps=norm_epsilon)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        ### mamba layer
        self.mamba_layers = nn.ModuleList(
            [
                create_block(
                    ssm_embed_dims,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=ssm_inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(1)
            ]
        )


        
        
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

    def forward_features(self, x, inference_params=None):

        # block = getattr(self, f"block")
        # patch_embed = getattr(self, f"patch_embed")
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
        x = x.reshape(-1,(self.imgae_size//self.patch_size)*(self.imgae_size//self.patch_size), self.embed_dims).contiguous() # 16,4096,256
        for layer in self.layers:
            x = layer(x)   # 128 1024 256
        
        
        # x = x

        residual = None
        hidden_states = x.reshape(self.batch_size, -1, self.embed_dims)
       
        for idx, layer in enumerate(self.mamba_layers):
            
            
            hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        
        # return hidden_states[:, 0, :]

        hidden_states = hidden_states.view(int(x.size(0)/self.num_frame),x.size(1),-1,self.embed_dims)
        return hidden_states.mean(2)

    def forward(self, x):
        # x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(1))
        return x

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = SSM_Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block




@register_model
def spikformer_mlla_spik4(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=128, img_size_w=128,patch_size=8, embed_dims=256, num_heads=8 , mlp_ratios=4,
        in_channels=3, num_classes=300, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2 , sr_ratios=1,num_frames=8,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import os
    import os


    model = Spikformer(
        img_size_h=128, img_size_w=128,patch_size=8, embed_dims=256, num_heads=8, mlp_ratios=4.0,
        in_channels=3, num_classes=300, qkv_bias=True, num_frames=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1, batch_size = 32
        # **kwargs
    ).cuda()
    

    # model = nn.DataParallel(model).cuda()
    # a = torch.rand(50, 8, 3, 128, 128).cuda(1)
    a = torch.rand(32, 3, 8, 128, 128).cuda()
    out = model.forward(a)

    print(out.shape)


