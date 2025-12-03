# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Dense self-attention at 7×7 resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification (10 digits for MNIST)

import fire                       # CLI helper: lets you run 'python file.py --arg=value'
import torch                      # Main PyTorch framework
import torch.nn as nn              # Neural network base classes
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.optim as optim        # Optimizers (AdamW, SGD, etc.)
import warp as wp                  # NVIDIA Warp JIT backend (for GPU kernel acceleration)
from torch import Tensor
from torch.optim.lr_scheduler import StepLR  # Reduces learning rate on schedule
from torchvision import datasets, transforms # Built-in datasets + preprocessing

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool       # Sparse pooling op
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution
from warpconvnet.nn.modules.sequential import Sequential                # Ordered list of sparse modules
from warpconvnet.nn.modules.activations import ReLU                     # Sparse-aware ReLU activation
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features

# ---------------------------------------------------------------------------
# Building blocks: small modular components used to construct the main model
# ---------------------------------------------------------------------------

class ConvBlock2d(Sequential):
    """
    Sparse 2D convolutional block base on WarpConvNet functions.
    Composition: SparseConv2d -> BatchNorm1d -> ReLU
    - relu activation needs to be disabled in some cases
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False, relu=True):
        super().__init__(
            SparseConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm1d(out_ch),
            ReLU(inplace=True) if relu is True else nn.Identity(),
        )    

# ---------------------------------------------------------------------------

class ConvTrBlock2d(nn.Module):
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

class ResidualSparseBlock2d(nn.Module):
    """
    Sparse residual block (the core computation unit of the encoder/decoder).
    This is the ResNet "BasicBlock" from mink_unet.py:
        Conv → BN → ReLU
        Conv → BN
        Add residual
        ReLU 
    - Sparse convolution layers based on ConvBlock2d 
    - 'relu=False' makes the second layer without activation
    - Skip connection: adds input ('identity') to output ('out')
    - Preserves sparse coordinate structure (no densification)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # if you need to downsample, the skip connection must downsample the input
        # before adding it to the output right before the last ReLU
        # we do this with a simple convolution block
        self.downsample = None
        if in_ch != out_ch:
            # Projection for channel mismatch between input/output
            self.downsample = ConvBlock2d(in_ch, out_ch, kernel_size=1, stride=1, relu=False)

        # First convolution: SparseConv2d + BatchNorm1d + ReLU
        self.conv1 = ConvBlock2d(in_ch, out_ch, kernel_size=3, stride=1)

        # Second convolution: SparseConv2d + BatchNorm1d
        self.conv2 = ConvBlock2d(out_ch, out_ch, kernel_size=3, stride=1, relu=False)

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

class BottleneckAttention2D(nn.Module):
    """
    FIXME FIXME: this is currently dense... look for sparse attention modules in WarpConvNet???
    Dense multi-head self-attention at the bottleneck.
    - Operates on small 7×7 dense feature maps (cost is negligible).
    - Provides global context by letting each spatial token attend to all others.
    """
    def __init__(self, channels: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True, dropout=dropout
        )
        hidden = int(channels * mlp_ratio)
        # A small MLP (2-layer 1×1 conv) adds non-linear mixing after attention
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
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class Net(nn.Module):
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
        self.enc1 = ResidualSparseBlock2d(1, 32)   # [B,1,28,28] → [B,32,28,28]
        self.enc2 = ResidualSparseBlock2d(32, 64)  # [B,32,14,14] → [B,64,14,14]

        # ---- Bottleneck (dense) ----
        # Feature mixing between encoder and decoder:
        # Dense attention expands channels, applies global MHSA, compresses back
        self.pre_attn  = nn.Conv2d(64, 128, kernel_size=1)      # channel lift 64→128
        self.attn      = BottleneckAttention2D(128, heads=4)    # global attention
        self.post_attn = nn.Conv2d(128, 64, kernel_size=1)      # back to 64 channels

        # ---- Decoder (sparse) ----
        # Symmetric to encoder but with upsampling and skip merges
        self.up1  = ConvTrBlock2d(64, 64, kernel_size=2, stride=2)  # upsample 7→14
        self.dec1 = ResidualSparseBlock2d(64 + 64, 64)              # merge skip2 + current

        self.up0  = ConvTrBlock2d(64, 32, kernel_size=2, stride=2)  # upsample 14→28
        self.dec0 = ResidualSparseBlock2d(32 + 32, 32)              # merge skip1 + current

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
        # --- Convert dense input image to sparse voxel representation
        xs = Voxels.from_dense(x)

        # ------------------ Encoder ------------------

        ### NOTE: this is using fixed downsampling (nothing is learned there)
        ### in mink_unet, downsamplig is done by an initial convolution w/ stride=2
        ### the resnet module (conv + norm + relu, etc..)

        xs = self.enc1(xs)                             # Sparse convs 1→32 @28×28
        skip1 = xs                                     # Save skip connection (28×28)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 14×14

        xs = self.enc2(xs)                             # Sparse convs 32→64 @14×14
        skip2 = xs                                     # Save skip (14×14)
        xs = sparse_max_pool(xs, kernel_size=(2,2), stride=(2,2))  # Downsample to 7×7

        # ------------------ Bottleneck ------------------
        bot_sparse = xs                                # Preserve stride & coords
        # Convert sparse features to dense for attention
        x_dense = xs.to_dense(channel_dim=1, spatial_shape=(7, 7))
        x_dense = self.pre_attn(x_dense)               # 64→128 channels
        x_dense = self.attn(x_dense)                   # global context
        x_dense = self.post_attn(x_dense)              # 128→64 channels
        # Convert back to sparse using same geometry (coords + stride)
        xs = Voxels.from_dense(x_dense, dense_tensor_channel_dim=1,
                               target_spatial_sparse_tensor=bot_sparse)

        # ------------------ Decoder ------------------
        # Upsample and fuse with sparse skip connections
        y = self.up1(xs, skip2)                        # 7→14 upsample
        y = cat(y, skip2)                              # sparse concatenation (channels: 64+64)
        y = self.dec1(y)                               # residual processing

        y = self.up0(y, skip1)                         # 14→28 upsample
        y = cat(y, skip1)                              # concat (32+32)
        y = self.dec0(y)                               # residual processing

        # ------------------ Head ------------------
        # Convert final sparse map back to dense for classification
        y_dense = y.to_dense(channel_dim=1, spatial_shape=(28,28))
        logits  = self.head(y_dense)
        return F.log_softmax(logits, dim=1)            # Log-probabilities for 10 digits


# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------

def train(model, device, train_loader, optimizer, epoch):
    """Single-epoch training loop."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)                      # Forward pass
        loss = F.nll_loss(output, target)         # Classification loss
        loss.backward()                           # Backpropagation
        optimizer.step()                          # Weight update

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")


def test(model, device, test_loader):
    """Evaluation loop (no gradient updates)."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test: Avg loss={test_loss:.4f}, Acc={acc:.2f}%")
    return acc


def main(
    batch_size=128,
    test_batch_size=1000,
    epochs=5,
    lr=1e-3,
    scheduler_step_size=10,
    gamma=0.7,
    device="cuda",
):
    """Main training driver."""
    wp.init()  # Initialize Warp backend
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    torch.manual_seed(1)

    # --- Data loading ---
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=test_batch_size, shuffle=True)

    # --- Model, optimizer, LR scheduler ---
    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        scheduler.step()
    print(f"Final accuracy: {acc:.2f}%")


if __name__ == "__main__":
    # Fire allows CLI usage, e.g.:
    #   python uresnet_sparse.py --epochs=10 --lr=0.001 --device=cuda
    fire.Fire(main)
