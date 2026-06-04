import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1 image, 3 channels, 20x20
x = torch.randn(1, 3, 20, 20)

# Keep stride 2; use padding=1 if you want a more comparable size
m = nn.Conv2d(3, 3, 2, stride=2, padding=1)

y = m(x)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(x[0].permute(1, 2, 0).detach().numpy())
axes[0].set_title("Original x")
axes[0].axis("off")

axes[1].imshow(y[0, 0].detach().numpy())
axes[1].set_title("Convolution output")
axes[1].axis("off")

plt.tight_layout()
plt.show()