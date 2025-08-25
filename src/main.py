import os
import sys
import argparse
import yaml

# Ensure project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    test_quick,
    IMAGES_DIR,
)
from src.evaluate import evaluate_model_from_ckpt
from src.preprocess import generate_synthetic_preview


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="REMEDy minimal experiments")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config", "config.yaml"), help="Path to YAML config")
    parser.add_argument("--experiment", type=str, default=None, choices=[None, "quick", "exp1", "exp2", "exp3", "eval"], help="Which experiment to run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = args.experiment or cfg.get("experiment", "quick")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if exp == "quick":
        print("[Main] Running quick end-to-end test...")
        test_quick()
    elif exp == "exp1":
        patterns = cfg.get("data", {}).get("patterns", ["blobs", "checkerboard", "stripes"])[:3]
        image_size = int(cfg.get("image_size", 32))
        steps = int(cfg.get("steps", 60))
        batch_size = int(cfg.get("batch_size", 32))
        run_experiment_1(patterns=patterns, image_size=image_size, steps=steps, batch_size=batch_size, out_images_dir=IMAGES_DIR)
    elif exp == "exp2":
        image_size = int(cfg.get("image_size", 32))
        steps_train = int(cfg.get("steps", 60))
        batch_size = int(cfg.get("batch_size", 32))
        run_experiment_2(image_size=image_size, steps_train=steps_train, batch_size=batch_size, out_images_dir=IMAGES_DIR)
    elif exp == "exp3":
        image_size = int(cfg.get("image_size", 32))
        run_experiment_3(image_size=image_size, out_images_dir=IMAGES_DIR)
    elif exp == "eval":
        image_size = int(cfg.get("image_size", 32))
        steps = int(cfg.get("steps", 20))
        router_thresh = float(cfg.get("router_thresh", 0.7))
        # Evaluate all three types if present
        for mt in ["adm", "revunet", "remedy"]:
            evaluate_model_from_ckpt(mt, image_size=image_size, steps=steps, router_thresh=router_thresh)
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    # Optionally generate synthetic previews
    if cfg.get("previews", True):
        print("[Main] Generating synthetic dataset previews...")
        generate_synthetic_preview(patterns=cfg.get("data", {}).get("patterns", ["blobs", "checkerboard", "stripes"]),
                                   size=int(cfg.get("image_size", 32)), n=9)


if __name__ == "__main__":
    main()
