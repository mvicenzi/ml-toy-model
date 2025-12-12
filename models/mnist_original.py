# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# This example trains a simple 2D sparse convolutional network (WarpConvNet)
# on the MNIST handwritten digits dataset.
#
# It demonstrates:
#   - Loading MNIST (28x28 grayscale digits)
#   - Defining a sparse 2D CNN using WarpConvNet modules
#   - Training and evaluating with PyTorch
#   - Using Warp (NVIDIA's JIT-compiled GPU kernels) for acceleration
# ---------------------------------------------------------------------------

import torch                      # PyTorch main library
import torch.nn as nn              # Neural network layers and base classes
import torch.nn.functional as F    # Functional (stateless) ops like ReLU, log_softmax, etc.
from torch import Tensor

# Import WarpConvNet-specific modules
import warpconvnet.nn.functional.transforms as T
from warpconvnet.geometry.types.voxels import Voxels           # Sparse voxel representation
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool  # Sparse max pooling
from warpconvnet.nn.modules.sequential import Sequential        # Sequential wrapper for sparse layers
from warpconvnet.nn.modules.sparse_conv import SparseConv2d     # Sparse 2D convolution layer


# -----------------------------------------------------------------------------
# Define a simple 2-layer sparse convolutional network
# -----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # NOTE: Must use warpconvnet.nn.modules.Sequential, not torch.nn.Sequential,
        # because this version is designed to handle *sparse voxel inputs*.
        # Each SparseConv2d layer only operates where there are non-zero voxels.

        self.layers = Sequential(
            SparseConv2d(1, 32, kernel_size=3, stride=1),  # Input: 1 channel -> 32 feature maps
            nn.ReLU(),                                     # Non-linearity
            SparseConv2d(32, 64, kernel_size=3, stride=1), # 32 -> 64 channels
            nn.ReLU(),
        )

        # After 2 convolutions and one 2x2 pool, the 28x28 image becomes roughly 14x14
        # Flattened feature vector length: 14*14*64 = 12544
        # Then a simple MLP ("head") for classification into 10 digit classes (0–9)
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 output classes
        )

    def forward(self, x: Tensor):
        # Convert dense image tensor (B,1,28,28) -> sparse voxel grid
        x = Voxels.from_dense(x)

        # Apply sparse convolutions and activations
        x = self.layers(x)

        # Downsample spatial resolution (like MaxPool2d, but for sparse data)
        x = sparse_max_pool(x, kernel_size=(2, 2), stride=(2, 2))

        # Convert sparse representation back to dense tensor
        # channel_dim=1 keeps channels in standard PyTorch layout: [B, C, H, W]
        # spatial_shape=(14,14) defines the spatial size after pooling
        x = x.to_dense(channel_dim=1, spatial_shape=(14, 14))

        # Flatten all spatial features for fully connected layers
        x = torch.flatten(x, 1)

        # Pass through the dense head
        x = self.head(x)

        # Log-softmax for classification (used with NLL loss)
        output = F.log_softmax(x, dim=1)
        return output