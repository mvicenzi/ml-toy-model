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

import fire                       # Simple CLI tool for running main() from terminal
import torch                      # PyTorch main library
import torch.nn as nn              # Neural network layers and base classes
import torch.nn.functional as F    # Functional (stateless) ops like ReLU, log_softmax, etc.
import torch.optim as optim        # Optimization algorithms (SGD, Adam, etc.)
import warp as wp                  # NVIDIA Warp framework (compiles GPU kernels just-in-time)
from torch import Tensor
from torch.optim.lr_scheduler import StepLR  # Scheduler to reduce LR periodically
from torchvision import datasets, transforms # Built-in dataset and preprocessing utilities

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


# -----------------------------------------------------------------------------
# Training function (one epoch)
# -----------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # set model to training mode (enables dropout, etc.)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()         # reset gradient buffers
        output = model(data)          # forward pass through network
        loss = F.nll_loss(output, target)  # negative log-likelihood loss (matches log_softmax)
        loss.backward()               # backprop gradients
        optimizer.step()              # update weights

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


# -----------------------------------------------------------------------------
# Evaluation / testing loop (no gradient updates)
# -----------------------------------------------------------------------------
def test(model, device, test_loader):
    model.eval()  # evaluation mode (disables dropout)
    test_loss = 0
    correct = 0
    with torch.no_grad():  # disables autograd for speed and memory
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum of all sample losses
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # predicted class = index with max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average loss and accuracy for reporting
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


# -----------------------------------------------------------------------------
# Main training driver (sets up everything)
# -----------------------------------------------------------------------------
def main(
    batch_size: int = 128,
    test_batch_size: int = 1000,
    epochs: int = 10,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
):
    # Initialize NVIDIA Warp (compiles GPU kernels / backend)
    wp.init()

    # Select device: GPU if available, else CPU
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Fix random seed for reproducibility
    torch.manual_seed(1)

    # -------------------------------------------------------------------------
    # Load MNIST dataset
    # -------------------------------------------------------------------------
    # Each image is 28x28 grayscale, label ∈ [0–9]
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",                     # dataset path
            train=True,                    # training split
            download=True,                 # download if not present
            transform=transforms.Compose([transforms.ToTensor()]),  # normalize to [0,1]
        ),
        batch_size=batch_size,
        shuffle=True,                      # randomize order each epoch
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,                   # test split
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )

    # -------------------------------------------------------------------------
    # Model, optimizer, and learning rate scheduler
    # -------------------------------------------------------------------------
    model = Net().to(device)                        # create model and send to GPU/CPU
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW optimizer (Adam + weight decay)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)
    # -> reduces lr by factor `gamma` every `scheduler_step_size` epochs

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()  # update LR schedule

    print(f"Final accuracy: {accuracy:.2f}%")


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Fire automatically exposes `main` arguments as command-line flags
    # Example run:
    #   python mnist_sparse.py --epochs=10 --lr=0.001 --device=cuda
    fire.Fire(main)

