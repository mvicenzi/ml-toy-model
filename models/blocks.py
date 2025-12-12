from torch import Tensor
import torch.nn as nn              # Neural network base classes

from warpconvnet.geometry.base.geometry import Geometry                 # Voxels derives from this
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution
from warpconvnet.nn.modules.sequential import Sequential                # Ordered list of sparse modules
from warpconvnet.nn.modules.activations import ReLU                     # Sparse-aware ReLU activation
from warpconvnet.nn.modules.normalizations import LayerNorm             # layer normalization 
from warpconvnet.nn.modules.attention import PatchAttention, SpatialFeatureAttention  # Sparse attention
from warpconvnet.nn.modules.activations import GELU                     # Sparse-aware ReLU activation

# ---------------------------------------------------------------------------
# Building blocks: small modular components used to construct the main model
# ---------------------------------------------------------------------------

class ConvBlock2D(Sequential):
    """
    Sparse 2D convolutional block based on WarpConvNet functions.
    Composition: 
        SparseConv2d -> BatchNorm1d -> ReLU
    - this is the main conv layer in base resnet block  
    - note: relu activation needs to be disabled in some cases!
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False, relu=True):
        super().__init__(
            SparseConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm1d(out_ch),
            ReLU(inplace=True) if relu is True else nn.Identity(),
        )  


# ---------------------------------------------------------------------------

class ConvTrBlock2D(nn.Module):
    """
    Sparse transposed convolution (upsampling block).
    Used in the decoder to increase spatial resolution.
    - 'transposed=True' performs the reverse of a convolution (learned upsampling).
    - out_spatial_sparsity defines which coordinates the upsampled result should align to.
    - relu activation always active
    """
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, bias=False):
        super().__init__()
        self.deconv = SparseConv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            transposed=True, bias=bias
        )
        self.norm_act = Sequential(
            nn.BatchNorm1d(out_ch),
            ReLU(inplace=True),
        )

    def forward(self, x_sparse: Voxels, out_spatial_sparsity: Voxels) -> Voxels:
        # Perform sparse transposed convolution guided by the skip tensor geometry
        y = self.deconv(x_sparse, out_spatial_sparsity)
        return self.norm_act(y)

# ---------------------------------------------------------------------------

class ResidualSparseBlock2D(nn.Module):
    """
    Sparse residual block (the core computation unit of the encoder/decoder).
    This is the ResNet "BasicBlock" from mink_unet.py:
        Conv → BN → ReLU
        Conv → BN
        Add residual
        ReLU 
    - Sparse convolution layers based on ConvBlock2D 
    - "stride" parameter always at 1: size downsampling is external!
    - 'relu=False' makes the second layer without activation
    - Skip connection: adds input ('identity') to output ('out')
    - Preserves sparse coordinate structure (no densification)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # if you downsample or add feature dimensions to pixels
        # input must be adapted to properly be added to output in residual skip connection
        # we do this with a simple convolution block
        self.downsample = None
        if stride!= 1 or in_ch != out_ch:
            self.downsample = ConvBlock2D(in_ch, out_ch, kernel_size=1, stride=stride, relu=False)

        # First convolution: SparseConv2d + BatchNorm1d + ReLU
        self.conv1 = ConvBlock2D(in_ch, out_ch, kernel_size=3, stride=stride)

        # Second convolution: SparseConv2d + BatchNorm1d
        self.conv2 = ConvBlock2D(out_ch, out_ch, kernel_size=3, stride=1, relu=False)

        # Final activation (after skip addition)
        self.act = ReLU(inplace=True)


    def forward(self, x_sparse: Voxels) -> Voxels:

        identity = x_sparse 
        if self.downsample is not None:
            identity = self.downsample(x_sparse)

        # Forward through two sparse conv layers
        out = self.conv1(x_sparse)
        out = self.conv2(out)

        out += identity  # Residual skip addition (still sparse)
        out = self.act(out)

        return out
    
# ---------------------------------------------------------------------------
# Bottleneck Attention block (dense)
# ---------------------------------------------------------------------------

class BottleneckDenseAttention2D(nn.Module):
    """
    Dense multi-head self-attention at the bottleneck.
    - Operates on small 7x7 dense feature maps (cost is negligible).
    - Provides global context by letting each spatial token attend to all others.
    Flow: norm -> MHSA -> residual -> MLP -> residual.

    """
    def __init__(self, channels: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True, dropout=dropout
        )
        hidden = int(channels * mlp_ratio)
        # A small MLP (2-layer 1x1 conv) adds non-linear mixing after attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> flatten spatial dims -> [B, HW, C]
        b, c, h, w = x.shape
        x = self.norm(x)
        tokens = x.flatten(2).transpose(1, 2)     # Convert to token sequence [B, HW, C]
        attn_out, _ = self.mha(tokens, tokens, tokens)  # Global self-attention
        tokens = tokens + attn_out                # Residual connection
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = x + self.mlp(x)                       # Another residual connection via MLP
        return x


# ---------------------------------------------------------------------------
# Bottleneck Attention block (sparse)
# ---------------------------------------------------------------------------

class BottleneckSparseAttention2D(nn.Module):
    """
    Sparse transformer-style bottleneck using WarpConvNet's PatchAttention.
    Operates directly on Geometry (e.g. Voxels), so we never densify.
    Flow: norm -> PathAttention -> residual -> MLP -> residual.
    """
    def __init__(self, channels: int, attn_channels: int, heads: int = 4, patch_size: int = 64, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_ratio: float = 2.0):
        super().__init__()

        # project before attention (sparse 1x1 conv)
        self.pre_proj = SparseConv2d(channels, attn_channels, kernel_size=1)
        
        # with not batchNorm?
        self.norm1 = LayerNorm(attn_channels)
        self.norm2 = LayerNorm(attn_channels)

        ## WARNING: can't Use PatchAttention --> does not support 2D
        ## note if patch size > spatial dimension, then it is global
        ##self.attn = PatchAttention(dim=attn_channels,patch_size=patch_size,num_heads=heads,
        ##                              qkv_bias=True,attn_drop=attn_drop, proj_drop=proj_drop)
        
        # --- trying to avoid the wrapper and calling the function with no encoding
                # Sparse attention over features (no positional encoding → 2D-safe)
        self.attn = SpatialFeatureAttention(dim=attn_channels, num_heads=heads, qkv_bias=True, qk_scale=None,
            attn_drop=attn_drop, proj_drop=proj_drop, num_encoding_channels=32, encoding_range=1.0,
            use_encoding=False,      # <-- IMPORTANT: ignore coordinates, works in 2D
            enable_flash=True, use_batched_qkv=True)

        hidden = int(attn_channels * mlp_ratio)
        # A small MLP (2-layer 1x1 conv) adds non-linear mixing after attention
        self.mlp = Sequential(
            SparseConv2d(attn_channels, hidden, kernel_size=1),
            GELU(),
            SparseConv2d(hidden, attn_channels, kernel_size=1)
        )

        # project back after attention (sparse 1x1 conv)
        self.post_proj = SparseConv2d(attn_channels, channels, kernel_size=1)

    def forward(self, x: Geometry) -> Geometry:

        x2 = self.pre_proj(x)

        # Geometry in/out 
        h = self.attn(self.norm1(x2))
        x2 = x2 + h 

        # MLP sub-block
        h2 = self.mlp(self.norm2(x2)) 
        x2 = x2 + h2 

        x_out = self.post_proj(x2)
        return x_out
