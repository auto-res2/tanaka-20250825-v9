import os
import sys
from typing import List

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
})

from src.train import SyntheticPatternsDataset, IMAGES_DIR, DATA_DIR


def generate_synthetic_preview(patterns: List[str] = ["blobs", "checkerboard", "stripes"], size: int = 32, n: int = 8):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for p in patterns:
        ds = SyntheticPatternsDataset(pattern=p, size=size, length=max(16, n))
        imgs = torch.stack([ds[i] for i in range(n)], dim=0)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        # Save a grid manually (no torchvision dependency required here)
        rows = int(np.sqrt(n)); cols = int(np.ceil(n / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = np.array(axes).reshape(rows, cols)
        for i in range(rows*cols):
            r, c = divmod(i, cols)
            axes[r, c].axis('off')
            if i < n:
                axes[r, c].imshow(imgs[i].permute(1,2,0).numpy())
        plt.tight_layout(pad=0.1)
        out_pdf = os.path.join(IMAGES_DIR, f"preview_{p}_{size}.pdf")
        plt.savefig(out_pdf, bbox_inches="tight"); plt.close()
        print(f"Saved synthetic preview: {out_pdf}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[Preprocess] Generating synthetic previews...")
    generate_synthetic_preview(patterns=["blobs", "checkerboard", "stripes"], size=32, n=9)
    print("[Preprocess] Done.")


if __name__ == "__main__":
    main()
