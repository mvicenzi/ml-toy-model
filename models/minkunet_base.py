from torch import Tensor
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution

from .blocks import (
    ConvBlock2D, ConvTrBlock2D, 
    ResidualSparseBlock2D    
    )

class MinkUNetBase(nn.Module):
    """
    U-ResNet-style sparse model following a trimmed MinkUNet18-style architecture.
    - Initial conv at full resolution
    - Encoder: strided convolutions + residual blocks (2 stages)
    - Bottleneck: extra residual block at 7x7 (no attention)
    - Decoder: transposed convolutions + skip connections + residual blocks (2 stages)
    - Head: dense classification layer (10 digits)

    Tensor flow (for 28x28 input):
    Encoder:   [B,1,28,28] -> [B,32,28,28] -> [B,32,14,14] -> [B,64,7,7]
    Bottleneck:            [B,64,7,7] -> [B,64,7,7]
    Decoder:   [B,64,7,7] -> [B,64,14,14] -> [B,64,28,28]
    """
    def __init__(self):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0 = ConvBlock2D(1, 32, kernel_size=3, stride=1)  # [B,1,28,28] -> [B,32,28,28]

        # ---- Encoder (2 stages) ----
        # Stage 1: 28x28 to 14x14
        self.conv1  = ConvBlock2D(32, 32, kernel_size=2, stride=2)    # Spatial downsample
        self.block1 = ResidualSparseBlock2D(32, 32)                   # Channels stay 32

        # Stage 2: 14x14 to 7x7
        self.conv2  = ConvBlock2D(32, 32, kernel_size=2, stride=2)    # Spatial downsample
        self.block2 = ResidualSparseBlock2D(32, 64)                   # 32 -> 64 channels

        # ---- Bottleneck (no attention, just a residual block at 7x7) ----
        # This plays the role of the bottom-stage blocks in MinkUNetBase.
        self.bottleneck = ResidualSparseBlock2D(64, 64)

        # ---- Decoder (2 stages, symmetric to encoder) ----
        # Stage 1: 7x7 -> 14x14
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block6  = ResidualSparseBlock2D(64 + 32, 64)              # Merge skip1, process

        # Stage 2: 14x14 -> 28x28 (full resolution)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # Upsample
        self.block8  = ResidualSparseBlock2D(64 + 32, 64)              # Merge skip0, process

        # ---- Final projection + classification head ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)    # Feature refinement
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

        # Initial conv at full resolution
        out = self.conv0(xs)               # [B,1,28,28] -> [B,32,28,28]
        out_p1 = out                       # Skip for final decoder stage

        # Stage 1: 28x28 -> 14x14
        out = self.conv1(out_p1)           # Downsample
        out = self.block1(out)             # Residual processing
        out_b1p2 = out                     # Skip at 14x14

        # Stage 2: 14x14 -> 7x7
        out = self.conv2(out_b1p2)         # Downsample
        out = self.block2(out)             # Residual + 32->64
        # Now out is [B,64,7,7] (sparse)

        # ============ BOTTLENECK (no attention) ============
        out = self.bottleneck(out)         # Extra residual processing at 7x7

        # ============ DECODER ============

        # Stage 1: 7x7 -> 14x14
        out = self.convtr5(out, out_b1p2)  # Upsample guided by skip geometry
        out = cat(out, out_b1p2)           # [B,64,14,14] + [B,32,14,14] = [B,96,14,14]
        out = self.block6(out)             # -> [B,64,14,14]

        # Stage 2: 14x14 -> 28x28
        out = self.convtr7(out, out_p1)    # Upsample
        out = cat(out, out_p1)             # [B,64,28,28] + [B,32,28,28] = [B,96,28,28]
        out = self.block8(out)             # -> [B,64,28,28]

        # ============ FINAL PROJECTION + HEAD ============

        out = self.final(out)              # [B,64,28,28]

        out_dense = out.to_dense(channel_dim=1, spatial_shape=(28, 28))
        logits = self.head(out_dense)
        return F.log_softmax(logits, dim=1)
