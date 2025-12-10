import fire                       # CLI helper: lets you run 'python file.py --arg=value'
import torch                      # Main PyTorch framework
import torch.optim as optim        # Optimizers (AdamW, SGD, etc.)
import torch.nn.functional as F    # Functional layer calls (stateless)
import warp as wp                  # NVIDIA Warp JIT backend (for GPU kernel acceleration)
from torch.optim.lr_scheduler import StepLR  # Reduces learning rate on schedule
from torchvision import datasets, transforms # Built-in datasets + preprocessing

from models import MODEL_REGISTRY

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
    model_name="mink_unet",
    batch_size=128,
    test_batch_size=1000,
    epochs=2,
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
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelCls = MODEL_REGISTRY[model_name]
    model = ModelCls().to(device)

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
    #   python training.py --epochs=10 --lr=0.001 --device=cuda
    fire.Fire(main)
