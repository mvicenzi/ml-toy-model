# ----------------------------- U-Res + Attention -----------------------------
# Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
# Bottleneck: Dense self-attention over 7x7 tokens (global context)
# Decoder:   Dense upsampling with U-Net skip connections + residual conv blocks
# Head:      Global average pool -> Linear(10) -> log_softmax

import fire                       # Simple CLI tool for running main() from terminal
import torch                      # PyTorch main library
import torch.nn as nn              # Neural network layers and base classes
import torch.nn.functional as F    # Functional (stateless) ops like ReLU, log_softmax, etc.
import torch.optim as optim        # Optimization algorithms (SGD, Adam, etc.)
import warp as wp                  # NVIDIA Warp framework (compiles GPU kernels just-in-time)
from torch import Tensor
from torch.optim.lr_scheduler import StepLR  # Scheduler to reduce LR periodically
from torchvision import datasets, transforms # Built-in dataset and preprocessing utilities

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool
from warpconvnet.nn.modules.sparse_conv import SparseConv2d
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.activations import ReLU

# -------------------------- dense residual building block --------------------
class ResidualDenseBlock(nn.Module):
    """Standard 2×(Conv-BN-ReLU) residual block for DENSE tensors."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        return F.relu(out, inplace=True)


# ------------------------- sparse residual building block --------------------
class ResidualSparseBlock(nn.Module):
    """
    Two SparseConv2d (+ReLU) with a 1×1 sparse projection if channels change.
    Operates on WarpConvNet sparse voxel tensors.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.need_proj = (in_ch != out_ch)
        self.conv1 = SparseConv2d(in_ch, out_ch, kernel_size=k, stride=1)
        self.conv2 = SparseConv2d(out_ch, out_ch, kernel_size=k, stride=1)
        self.proj  = SparseConv2d(in_ch, out_ch, kernel_size=1, stride=1) if self.need_proj else None
        self.act   = ReLU()

    def forward(self, x_sparse):
        identity = x_sparse
        out = self.act(self.conv1(x_sparse))
        out = self.conv2(out)
        if self.need_proj:
            identity = self.proj(identity)
        out = out + identity
        return self.act(out)


# ------------------------------ attention block ------------------------------
class BottleneckAttention(nn.Module):
    """
    Multi-Head Self-Attention over flattened (H*W) tokens.
    Expects/returns DENSE tensors shaped [B, C, H, W].
    """
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.mha  = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,C,H,W] -> tokens [B, HW, C]
        b, c, h, w = x.shape
        x = self.norm(x)
        tokens = x.flatten(2).transpose(1, 2)        # [B, HW, C]
        attn_out, _ = self.mha(tokens, tokens, tokens)  # self-attention
        tokens = tokens + attn_out                     # residual
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = x + self.mlp(x)                            # residual MLP
        return x


# ---------------------------------- model ------------------------------------
class Net(nn.Module):
    """
    U-Res CNN with sparse encoder and dense decoder, plus an attention bottleneck.
    Designed for 28×28 grayscale inputs (MNIST).
    """
    def __init__(self):
        super().__init__()

        # -------- Encoder (sparse) --------
        # Stage E1: 28x28, channels 1 -> 32
        self.enc1 = ResidualSparseBlock(1, 32)
        # Pool -> 14x14
        # Stage E2: 14x14, channels 32 -> 64
        self.enc2 = ResidualSparseBlock(32, 64)
        # Pool -> 7x7

        # -------- Channel mixers at the bottleneck (dense) --------
        # Convert to dense at 7x7, lift channels, attention, then project back.
        self.pre_attn  = nn.Conv2d(64, 128, kernel_size=1)
        self.attn      = BottleneckAttention(128, num_heads=4, mlp_ratio=2.0)
        self.post_attn = nn.Conv2d(128, 64, kernel_size=1)

        # -------- Decoder (dense, U-Net style) --------
        # Up 7->14, fuse skip from E2 (64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 7->14
        self.dec1 = ResidualDenseBlock(64 + 64, 64)  # concat skip: cat([up, skip2], dim=1)

        # Up 14->28, fuse skip from E1 (32)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 14->28
        self.dec0 = ResidualDenseBlock(32 + 32, 32)

        # -------- Head (classification) --------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pool -> [B, C, 1, 1]
            nn.Flatten(),             # [B, C]
            nn.Linear(32, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: dense input [B, 1, 28, 28]
        """
        # ---- to sparse for the encoder ----
        xs = Voxels.from_dense(x)          # sparse @ 28x28

        # E1: sparse residual block @28x28
        xs = self.enc1(xs)

        # Save skip1 as DENSE (28x28, C=32) before pooling
        skip1 = xs.to_dense(channel_dim=1, spatial_shape=(28, 28))

        # Pool to 14x14 (sparse)
        xs = sparse_max_pool(xs, kernel_size=(2, 2), stride=(2, 2))

        # E2: sparse residual block @14x14
        xs = self.enc2(xs)

        # Save skip2 as DENSE (14x14, C=64) before pooling
        skip2 = xs.to_dense(channel_dim=1, spatial_shape=(14, 14))

        # Pool to 7x7 (sparse)
        xs = sparse_max_pool(xs, kernel_size=(2, 2), stride=(2, 2))

        # ---- bottleneck attention in dense ----
        x_dense = xs.to_dense(channel_dim=1, spatial_shape=(7, 7))  # [B, 64, 7, 7]
        x_dense = self.pre_attn(x_dense)                            # [B, 128, 7, 7]
        x_dense = self.attn(x_dense)                                # MHSA over 7x7 tokens
        x_dense = self.post_attn(x_dense)                           # back to 64 ch

        # ---- decoder with U-Net skips (dense) ----
        # Up to 14x14 and fuse with skip2
        y = self.up1(x_dense)                                       # [B, 64, 14, 14]
        y = torch.cat([y, skip2], dim=1)                            # [B, 128, 14, 14]
        y = self.dec1(y)                                            # [B, 64, 14, 14]

        # Up to 28x28 and fuse with skip1
        y = self.up0(y)                                             # [B, 32, 28, 28]
        y = torch.cat([y, skip1], dim=1)                            # [B, 64, 28, 28]
        y = self.dec0(y)                                            # [B, 32, 28, 28]

        # ---- classification head ----
        logits = self.head(y)                                       # [B, 10]
        return F.log_softmax(logits, dim=1)



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
            "./data",                     # dataset path
            train=True,                    # training split
            download=True,                 # download if not present
            transform=transforms.Compose([transforms.ToTensor()]),  # normalize to [0,1]
        ),
        batch_size=batch_size,
        shuffle=True,                      # randomize order each epoch
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
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
    #   python uresnet.py --epochs=10 --lr=0.001 --device=cuda
    fire.Fire(main)

