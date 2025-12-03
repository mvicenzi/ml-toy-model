import torch, matplotlib.pyplot as plt
from torchvision import datasets, transforms

# MNIST: 28×28 grayscale digits
transform = transforms.ToTensor()
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# quick visual check
fig, axs = plt.subplots(6, 12, figsize=(10,12))
axes = axs.flatten()
for i in range(len(axes)):
    img, y = train[i]
    axes[i].imshow(img.squeeze(0), cmap="gray")
    axes[i].set_title(y); axes[i].axis("off")

plt.tight_layout()
plt.show()

