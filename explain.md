## Overview

The model combines sparse convolutional encoding (for efficiency), a dense attention bottleneck (for global context), and a dense decoder with U-Net–style skip connections (for spatial detail). It is designed for classifying 28×28 grayscale MNIST images.

---

## Data

**Input shape:** `[B, 1, 28, 28]` — a batch of B grayscale images, each 28×28 pixels.
**Output:** 10 log-probabilities (digits 0–9).

---

## Architecture Summary

### Encoder (Sparse)

* **Goal:** Extract features while reducing spatial size.
* **Layers:**

  1. `ResidualSparseBlock(1→32)` at 28×28
  2. `sparse_max_pool` (down to 14×14)
  3. `ResidualSparseBlock(32→64)` at 14×14
  4. `sparse_max_pool` (down to 7×7)

The encoder processes only nonzero pixels using WarpConvNet’s `SparseConv2d`, improving efficiency on mostly empty images.

**Skip features:**

* `skip1` = dense copy after first block (32×28×28)
* `skip2` = dense copy after second block (64×14×14)

---

### Bottleneck (Dense Attention)

* Converts sparse features to dense at 7×7.
* **Layers:**

  1. `Conv2d(64→128)`
  2. `BottleneckAttention(128)` (multi-head self-attention + MLP)
  3. `Conv2d(128→64)`

This block captures long-range dependencies across the entire 7×7 feature map, enabling global reasoning about the image.

---

### Decoder (Dense, U-Net style)

* **Goal:** Reconstruct high-resolution features with skip connections.
* **Layers:**

  1. `ConvTranspose2d(64→64)` upsampling 7×7 → 14×14
     → concatenate with `skip2` → `ResidualDenseBlock(128→64)`
  2. `ConvTranspose2d(64→32)` upsampling 14×14 → 28×28
     → concatenate with `skip1` → `ResidualDenseBlock(64→32)`

Skip connections directly link encoder outputs to matching decoder levels, allowing recovery of spatial information lost during downsampling.

---

### Classification Head

* `AdaptiveAvgPool2d(1)` → `Flatten()` → `Linear(32→10)` → `log_softmax`
* Produces class probabilities over the ten digits.

---

## Building Blocks

| Block                   | Type   | Description                                                              |
| ----------------------- | ------ | ------------------------------------------------------------------------ |
| **ResidualSparseBlock** | Sparse | Two `SparseConv2d` + ReLU with residual connection.                      |
| **ResidualDenseBlock**  | Dense  | Two `Conv2d` + BatchNorm + ReLU with residual connection.                |
| **BottleneckAttention** | Dense  | Multi-head self-attention and MLP operating on flattened spatial tokens. |

---

## Data Flow Summary

| Stage      | Operation        | Channels   | Resolution |
| ---------- | ---------------- | ---------- | ---------- |
| Input      | –                | 1          | 28×28      |
| Encoder1   | Sparse residual  | 1→32       | 28×28      |
| Pool1      | Sparse max-pool  | –          | 14×14      |
| Encoder2   | Sparse residual  | 32→64      | 14×14      |
| Pool2      | Sparse max-pool  | –          | 7×7        |
| Bottleneck | Attention        | 64→128→64  | 7×7        |
| Decoder1   | Upsample + skip2 | (64+64)→64 | 14×14      |
| Decoder0   | Upsample + skip1 | (32+32)→32 | 28×28      |
| Head       | Global pool + FC | 32→10      | 1×1        |

