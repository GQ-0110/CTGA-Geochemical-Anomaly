"""Quick-test for C&G pre-review"""

import os
import sys
import torch
from vit_pytorch import Transformer
from moco.builder import MoCo_ViT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("device:", device)
    vit = Transformer(
        band_size=11,
        patch_size=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=64,
    )

    model = MoCo_ViT(vit, dim=64, mlp_dim=128, T=0.07, K=128, m=0.99).to(device)
    model.eval()

    x1 = torch.randn(16, 11, device=device)
    x2 = torch.randn(16, 11, device=device)

    with torch.no_grad():
        loss, logits, labels = model(x1, x2)

    print("Quick test successful.")

if __name__ == "__main__":
    main()
