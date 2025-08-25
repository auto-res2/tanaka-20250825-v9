import os
import sys
from typing import Dict, Tuple, Optional

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
})

from src.train import (
    device_auto,
    measure_peak_gpu_mem_and_time,
    GaussianDiffusion,
    TrainConfig,
    build_model,
    sample_ddim,
    IMAGES_DIR,
)

try:
    from torchvision.utils import make_grid
except Exception:
    make_grid = None

MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_samples(model: torch.nn.Module, diffusion: GaussianDiffusion, out_pdf: str, nrow: int = 4,
                 shape=(8, 3, 32, 32), steps: int = 20, router_thresh: Optional[float] = None):
    model.eval()
    with torch.no_grad():
        samples = sample_ddim(model, diffusion, shape=shape, steps=steps, router_thresh=router_thresh)
        samples = (samples.clamp(-1, 1) + 1) / 2.0  # to [0,1]
        samples = samples.float().cpu()
        if make_grid is not None:
            grid = make_grid(samples, nrow=nrow, padding=2)
            plt.figure(figsize=(nrow*2, (shape[0]//nrow)*2))
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.tight_layout(pad=0)
            plt.savefig(out_pdf, bbox_inches="tight")
            plt.close()
        else:
            # Fallback: save each image
            os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
            base, _ = os.path.splitext(out_pdf)
            for i, img in enumerate(samples):
                plt.figure(figsize=(2,2)); plt.axis('off')
                plt.imshow(img.permute(1,2,0).numpy())
                plt.tight_layout(pad=0)
                plt.savefig(f"{base}_{i:02d}.pdf", bbox_inches="tight"); plt.close()


def evaluate_model_from_ckpt(model_type: str, image_size: int = 32, steps: int = 20,
                              router_thresh: Optional[float] = 0.7) -> Dict:
    device = device_auto()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Build model and (optionally) load checkpoint if available
    default_cfgs = {
        "adm": TrainConfig(model_type="adm", with_attn=True),
        "revunet": TrainConfig(model_type="revunet", with_attn=True, use_ckpt=True),
        "remedy": TrainConfig(model_type="remedy", with_attn=True, tdq=True, compressor_bits=8, n_tgroups=8),
    }
    cfg = default_cfgs[model_type]
    model = build_model(cfg).to(device).to(dtype)
    ckpt_path = os.path.join(MODELS_DIR, f"{model_type}_tiny.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found, evaluating randomly initialized model: {ckpt_path}")

    diffusion = GaussianDiffusion(T=1000)

    # Measure latency
    def infer_once():
        _ = sample_ddim(model, diffusion, shape=(1, 3, image_size, image_size), steps=steps,
                        router_thresh=(router_thresh if model_type == 'remedy' else None))
    peak_inf_mb, avg_latency = measure_peak_gpu_mem_and_time(infer_once, warmup=1, iters=3)
    print(f"Eval {model_type}: peak GPU MB={peak_inf_mb if peak_inf_mb is not None else 'N/A (CPU)'}; latency={avg_latency:.4f}s")

    # Save sample grid
    out_pdf = os.path.join(IMAGES_DIR, f"samples_{model_type}_{image_size}.pdf")
    save_samples(model, diffusion, out_pdf, nrow=4, shape=(8, 3, image_size, image_size), steps=steps,
                 router_thresh=(router_thresh if model_type == 'remedy' else None))
    print(f"Saved samples: {out_pdf}")

    return {"peak_mb": peak_inf_mb, "latency": avg_latency, "samples_pdf": out_pdf}


if __name__ == "__main__":
    # Minimal manual test
    evaluate_model_from_ckpt("remedy", image_size=16, steps=10, router_thresh=0.7)
