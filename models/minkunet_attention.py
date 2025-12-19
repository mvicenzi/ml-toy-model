# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Dense self-attention at 7×7 resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification (10 digits for MNIST)

from torch import Tensor
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution

from .blocks import (
    ConvBlock2D, ConvTrBlock2D, 
    ResidualSparseBlock2D, 
    BottleneckDenseAttention2D,
    BottleneckSparseAttention2D
    )

# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class MinkUNetDenseAttention(nn.Module):
    """
    U-ResNet-style sparse model following MinkUNet18 architecture.
    - Initial conv at full resolution
    - Encoder: strided convolutions + residual blocks (2 stages)
    - Bottleneck: dense attention at 7×7 resolution
    - Decoder: transposed convolutions + skip connections + residual blocks (2 stages)
    - Head: dense classification layer (10 digits)
    
    Tensor flow (for 28×28 input):
    Encoder:  [B,1,28,28] → [B,32,28,28] → [B,32,14,14] → [B,64,7,7]
    Bottleneck: [B,64,7,7] → [B,128,7,7] (attention) → [B,64,7,7]
    Decoder:  [B,64,7,7] → [B,64,14,14] → [B,64,28,28]
    """
    def __init__(self):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0 = ConvBlock2D(1, 32, kernel_size=3, stride=1)  # [B,1,28,28] → [B,32,28,28]

        # ---- Encoder (2 stages) ----
        # Stage 1: 28×28 to 14×14
        self.conv1 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block1 = ResidualSparseBlock2D(32, 32)                # Channel stays 32
        
        # Stage 2: 14×14 to 7×7
        self.conv2 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block2 = ResidualSparseBlock2D(32, 64)                # Channel projection 32 to 64

        # ---- Bottleneck (dense attention at 7×7) ----
        # Global context at 7×7 resolution (49 spatial tokens)
        self.pre_attn  = nn.Conv2d(64, 128, kernel_size=1)      # channel lift 64 to 128
        self.attn      = BottleneckDenseAttention2D(128, heads=4)    # global attention
        self.post_attn = nn.Conv2d(128, 64, kernel_size=1)      # back to 64 channels

        # ---- Decoder (2 stages, symmetric to encoder) ----
        # Stage 1: 7×7 to 14×14
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block6 = ResidualSparseBlock2D(64 + 32, 64)               # Merge skip1, process
        
        # Stage 2: 14×14 to 28×28 (full resolution)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block8 = ResidualSparseBlock2D(64 + 32, 64)               # Merge skip0, process

        # ---- Final projection + classification head ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)  # Feature refinement
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire network.
        Input: [B,1,28,28] dense tensor
        Output: [B,10] log-probabilities (digit classes)
        """
        # Convert dense input image to sparse voxel representation
        xs = Voxels.from_dense(x)

        # ============ ENCODER ============
        
        # Initial convolution at full resolution
        out = self.conv0(xs)                    # [B,1,28,28] to [B,32,28,28]
        out_p1 = out                            # Skip connection for final decoder stage
        
        # Stage 1: 28×28 to 14×14
        out = self.conv1(out_p1)                # Downsample spatially
        out = self.block1(out)                  # Residual processing
        out_b1p2 = out                          # Skip connection [B,32,14,14]
        
        # Stage 2: 14×14 to 7×7
        out = self.conv2(out_b1p2)              # Downsample spatially
        out = self.block2(out)                  # Residual + channel projection 32 to 64
                                                # Result: [B,64,7,7] 

        # ============ BOTTLENECK (Dense Attention at 7×7) ============
        # Global context with 49 spatial tokens (7×7)
        bot_sparse = out                        # Preserve stride & coords
        x_dense = out.to_dense(channel_dim=1, spatial_shape=(7, 7))
        x_dense = self.pre_attn(x_dense)        # 64 to 128 channels
        x_dense = self.attn(x_dense)            # Global attention at 7×7
        x_dense = self.post_attn(x_dense)       # 128 to 64 channels
        out = Voxels.from_dense(x_dense, dense_tensor_channel_dim=1,
                                target_spatial_sparse_tensor=bot_sparse)

        # ============ DECODER ============
        
        # Stage 1: 7×7 → 14×14
        out = self.convtr5(out, out_b1p2)       # Upsample, guided by skip geometry
        out = cat(out, out_b1p2)                # [B,64,14,14] + [B,32,14,14] = [B,96,14,14]
        out = self.block6(out)                  # Process to [B,64,14,14]
        
        # Stage 2: 14×14 to 28×28 (full resolution)
        out = self.convtr7(out, out_p1)         # Upsample
        out = cat(out, out_p1)                  # [B,64,28,28] + [B,32,28,28] = [B,96,28,28]
        out = self.block8(out)                  # Process to [B,64,28,28]

        # ============ FINAL PROJECTION + HEAD ============
        out = self.final(out)                   # Feature refinement [B,64,28,28]
        
        # Convert to dense for classification
        out_dense = out.to_dense(channel_dim=1, spatial_shape=(28, 28))
        logits = self.head(out_dense)           # Global pool + classify
        return F.log_softmax(logits, dim=1)     # Log-probabilities for 10 digits
    

# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class MinkUNetSparseAttention(nn.Module):
    """
    U-ResNet-style sparse model following MinkUNet18 architecture.
    - Initial conv at full resolution
    - Encoder: strided convolutions + residual blocks (2 stages)
    - Bottleneck: sparse attention at 7x7 resolution
    - Decoder: transposed convolutions + skip connections + residual blocks (2 stages)
    - Head: dense classification layer (10 digits)
    
    Tensor flow (for 28x28 input):
    Encoder:  [B,1,28,28] → [B,32,28,28] → [B,32,14,14] → [B,64,7,7]
    Bottleneck: [B,64,7,7] → [B,128,7,7] (attention) → [B,64,7,7]
    Decoder:  [B,64,7,7] → [B,64,14,14] → [B,64,28,28]
    """
    def __init__(self, *, spatial_encoding: bool = True, flash_attention: bool = True, **kwargs,):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0 = ConvBlock2D(1, 32, kernel_size=3, stride=1)  # [B,1,28,28] → [B,32,28,28]

        # ---- Encoder (2 stages) ----
        # Stage 1: 28×28 to 14×14
        self.conv1 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block1 = ResidualSparseBlock2D(32, 32)                # Channel stays 32
        
        # Stage 2: 14×14 to 7×7
        self.conv2 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block2 = ResidualSparseBlock2D(32, 64)                # Channel projection 32 to 64

        # ---- Bottleneck (dense attention at 7×7) ----
        # Global context at 7×7 resolution (49 spatial tokens)
        self.bottleneck = BottleneckSparseAttention2D(channels=64, attn_channels=128, heads=4, 
                                                      encoding=spatial_encoding, flash=flash_attention)

        # ---- Decoder (2 stages, symmetric to encoder) ----
        # Stage 1: 7×7 to 14×14
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block6 = ResidualSparseBlock2D(64 + 32, 64)               # Merge skip1, process
        
        # Stage 2: 14×14 to 28×28 (full resolution)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block8 = ResidualSparseBlock2D(64 + 32, 64)               # Merge skip0, process

        # ---- Final projection + classification head ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)  # Feature refinement
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire network.
        Input: [B,1,28,28] dense tensor
        Output: [B,10] log-probabilities (digit classes)
        """
        # Convert dense input image to sparse voxel representation
        xs = Voxels.from_dense(x)

        # ============ ENCODER ============
        
        # Initial convolution at full resolution
        out = self.conv0(xs)                    # [B,1,28,28] to [B,32,28,28]
        out_p1 = out                            # Skip connection for final decoder stage
        
        # Stage 1: 28×28 to 14×14
        out = self.conv1(out_p1)                # Downsample spatially
        out = self.block1(out)                  # Residual processing
        out_b1p2 = out                          # Skip connection [B,32,14,14]
        
        # Stage 2: 14×14 to 7×7
        out = self.conv2(out_b1p2)              # Downsample spatially
        out = self.block2(out)                  # Residual + channel projection 32 to 64
                                                # Result: [B,64,7,7] 

        # ============ BOTTLENECK (Sparse Attention at 7×7) ============
        out = self.bottleneck(out)              # [B,64,7,7] -> [B,128,7,7] (attention) -> [B,64,7,7]

        # ============ DECODER ============
        
        # Stage 1: 7×7 → 14×14
        out = self.convtr5(out, out_b1p2)       # Upsample, guided by skip geometry
        out = cat(out, out_b1p2)                # [B,64,14,14] + [B,32,14,14] = [B,96,14,14]
        out = self.block6(out)                  # Process to [B,64,14,14]
        
        # Stage 2: 14×14 to 28×28 (full resolution)
        out = self.convtr7(out, out_p1)         # Upsample
        out = cat(out, out_p1)                  # [B,64,28,28] + [B,32,28,28] = [B,96,28,28]
        out = self.block8(out)                  # Process to [B,64,28,28]

        # ============ FINAL PROJECTION + HEAD ============
        out = self.final(out)                   # Feature refinement [B,64,28,28]
        
        # Convert to dense for classification
        out_dense = out.to_dense(channel_dim=1, spatial_shape=(28, 28))
        logits = self.head(out_dense)           # Global pool + classify
        return F.log_softmax(logits, dim=1)     # Log-probabilities for 10 digits
    

class MinkUNetSparseAttentionNoEnc(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)

class MinkUNetSparseAttentionNoFlash(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)

class MinkUNetSparseAttentionNoFlashEnc(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=False, **kwargs)
