from torch import Tensor
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool       # Sparse pooling op
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features

from .blocks import (
    ConvTrBlock2D, 
    ResidualSparseBlock2D, 
    BottleneckDenseAttention2D
    )

# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# --------------------------------------------------------------------------

class UResNetSparsePool(nn.Module):
    """
    U-ResNet-style sparse model with an attention bottleneck.
    - Encoder: sparse convolutions and pooling
    - Bottleneck: residual sparse block with no attention
    - Decoder: sparse upsampling and skip connections
    - Head: dense classification layer (10 digits)
    """
    def __init__(self):
        super().__init__()

        # ---- Encoder (sparse) ----
        # Each stage halves spatial resolution via sparse pooling and doubles channels
        self.enc1 = ResidualSparseBlock2D(1, 32)   # [B,1,28,28] to [B,32,28,28]
        self.enc2 = ResidualSparseBlock2D(32, 64)  # [B,32,14,14] to [B,64,14,14]

        # ---- Bottleneck (no attention, just a residual block at 7x7) ----
        # This plays the role of the bottom-stage blocks in MinkUNetBase.
        self.bottleneck = ResidualSparseBlock2D(64, 64)

        # ---- Decoder (sparse) ----
        # Symmetric to encoder but with upsampling and skip merges
        self.up1  = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # upsample 7→14
        self.dec1 = ResidualSparseBlock2D(64 + 64, 64)              # merge skip2 + current

        self.up0  = ConvTrBlock2D(64, 32, kernel_size=2, stride=2)  # upsample 14→28
        self.dec0 = ResidualSparseBlock2D(32 + 32, 32)              # merge skip1 + current

        # ---- Head (dense classification) ----
        # Pool global information into a vector and output 10 logits
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire network.
        Input: [B,1,28,28] dense tensor
        Output: [B,10] log-probabilities (digit classes)
        """
        ### Convert dense input image to sparse voxel representation
        ### this needs to be done because MNIST images are not actually sparse 
        xs = Voxels.from_dense(x)

        # ------------------ Encoder ------------------

        ### NOTE: this is using fixed downsampling (nothing is learned there)
        ### in mink_unet, downsamplig is done by an initial convolution w/ stride=2
        ### the resnet encoding module (conv + norm + relu, etc..) is then stride=1

        ### also this is semplitfication from mink_unet
        xs = self.enc1(xs)                             # Sparse convs 1 to 32 @28×28
        skip1 = xs                                     # Save skip connection (28×28)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 14×14

        xs = self.enc2(xs)                             # Sparse convs 32 to 64 @14×14
        skip2 = xs                                     # Save skip (14×14)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 7×7

        # ------------------ Bottleneck ------------------
        xs = self.bottleneck(xs)

        # ------------------ Decoder ------------------
        # Upsample and fuse with sparse skip connections

        y = self.up1(xs, skip2)                        # 7 to 14 upsample
        y = cat(y, skip2)                              # sparse concatenation (channels: 64+64)
        y = self.dec1(y)                               # residual processing

        y = self.up0(y, skip1)                         # 14 to 28 upsample
        y = cat(y, skip1)                              # concat (32+32)
        y = self.dec0(y)                               # residual processing

        # ------------------ Head ------------------
        # Convert final sparse map back to dense for classification
        y_dense = y.to_dense(channel_dim=1, spatial_shape=(28,28))
        logits  = self.head(y_dense)
        return F.log_softmax(logits, dim=1)            # Log-probabilities for 10 digits


# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class UResNetSparsePoolAttention(nn.Module):
    """
    U-ResNet-style sparse model with an attention bottleneck.
    - Encoder: sparse convolutions and pooling
    - Bottleneck: dense attention for global context
    - Decoder: sparse upsampling and skip connections
    - Head: dense classification layer (10 digits)
    """
    def __init__(self):
        super().__init__()

        # ---- Encoder (sparse) ----
        # Each stage halves spatial resolution via sparse pooling and doubles channels
        self.enc1 = ResidualSparseBlock2D(1, 32)   # [B,1,28,28] to [B,32,28,28]
        self.enc2 = ResidualSparseBlock2D(32, 64)  # [B,32,14,14] to [B,64,14,14]

        # ---- Bottleneck ----
        # Feature mixing between encoder and decoder:
        # Dense attention expands channels, applies global MHSA, compresses back
        self.pre_attn  = nn.Conv2d(64, 128, kernel_size=1)      # channel lift 64→128
        self.attn      = BottleneckDenseAttention2D(128, heads=4)    # global attention
        self.post_attn = nn.Conv2d(128, 64, kernel_size=1)      # back to 64 channels

        # ---- Decoder (sparse) ----
        # Symmetric to encoder but with upsampling and skip merges
        self.up1  = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)  # upsample 7→14
        self.dec1 = ResidualSparseBlock2D(64 + 64, 64)              # merge skip2 + current

        self.up0  = ConvTrBlock2D(64, 32, kernel_size=2, stride=2)  # upsample 14→28
        self.dec0 = ResidualSparseBlock2D(32 + 32, 32)              # merge skip1 + current

        # ---- Head (dense classification) ----
        # Pool global information into a vector and output 10 logits
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire network.
        Input: [B,1,28,28] dense tensor
        Output: [B,10] log-probabilities (digit classes)
        """
        ### Convert dense input image to sparse voxel representation
        ### this needs to be done because MNIST images are not actually sparse 
        xs = Voxels.from_dense(x)

        # ------------------ Encoder ------------------

        ### NOTE: this is using fixed downsampling (nothing is learned there)
        ### in mink_unet, downsamplig is done by an initial convolution w/ stride=2
        ### the resnet encoding module (conv + norm + relu, etc..) is then stride=1

        ### also this is semplitfication from mink_unet
        xs = self.enc1(xs)                             # Sparse convs 1 to 32 @28×28
        skip1 = xs                                     # Save skip connection (28×28)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 14×14

        xs = self.enc2(xs)                             # Sparse convs 32 to 64 @14×14
        skip2 = xs                                     # Save skip (14×14)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 7×7

        # ------------------ Bottleneck ------------------
        ### this is going back to being dense
        ### this allows to call pytorch standard attention layer
        ### needs to find spare alternative?? 

        bot_sparse = xs                                # Preserve stride & coords
        # Convert sparse features to dense for attention
        x_dense = xs.to_dense(channel_dim=1, spatial_shape=(7, 7))
        x_dense = self.pre_attn(x_dense)               # 64 to 128 channels
        x_dense = self.attn(x_dense)                   # global context
        x_dense = self.post_attn(x_dense)              # 128 to 64 channels

        # Convert back to sparse using same geometry (coords + stride)
        xs = Voxels.from_dense(x_dense, dense_tensor_channel_dim=1,
                               target_spatial_sparse_tensor=bot_sparse)

        # ------------------ Decoder ------------------
        # Upsample and fuse with sparse skip connections

        y = self.up1(xs, skip2)                        # 7 to 14 upsample
        y = cat(y, skip2)                              # sparse concatenation (channels: 64+64)
        y = self.dec1(y)                               # residual processing

        y = self.up0(y, skip1)                         # 14 to 28 upsample
        y = cat(y, skip1)                              # concat (32+32)
        y = self.dec0(y)                               # residual processing

        # ------------------ Head ------------------
        # Convert final sparse map back to dense for classification
        y_dense = y.to_dense(channel_dim=1, spatial_shape=(28,28))
        logits  = self.head(y_dense)
        return F.log_softmax(logits, dim=1)            # Log-probabilities for 10 digits