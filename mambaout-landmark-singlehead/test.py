# test.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import get_centers_from_heatmap


def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def test_metrics_cc(model, loader, device, image_size=(256, 256), skip_empty_gt=True):
    """
    Unified numpy-based evaluation for both insertion points.

    GT and prediction centers are BOTH extracted with
    `get_centers_from_heatmap` (connected-component centroid of the peak),
    so these numbers are directly comparable to the training/validation curve.

    Reports:
    - Euclidean distance (pixels / mm)
    - MSE (pixels² / mm²)
    - Relative error and MRE (%)
    """
    model.eval()

    H, W = image_size
    diag = np.sqrt(H ** 2 + W ** 2)

    records = []

    for images, targets, spacings in tqdm(loader, desc="Testing (CC method)"):
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        outputs = model(images)

        B = images.shape[0]

        for i in range(B):
            row = {}
            spacing_x = spacings[i][0].item()
            spacing_y = spacings[i][1].item()

            for ch, key in enumerate(["p1", "p2"]):
                pred_heat = outputs[i, ch].float().cpu().numpy()
                gt_heat = targets[i, ch].float().cpu().numpy()

                # Skip empty GT slices: an all-zero heatmap has no real center.
                if skip_empty_gt and gt_heat.max() <= 0:
                    continue

                # --- unified extraction: same rule for GT and prediction ---
                gt_x, gt_y = get_centers_from_heatmap(gt_heat)
                pred_x, pred_y = get_centers_from_heatmap(pred_heat)

                dx = pred_x - gt_x
                dy = pred_y - gt_y
                dist_px = np.sqrt(dx ** 2 + dy ** 2)

                dx_mm = dx * spacing_x
                dy_mm = dy * spacing_y
                dist_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2)

                diff_px = np.array([dx, dy])
                mse_px = float(np.mean(diff_px ** 2))

                diff_mm = np.array([dx_mm, dy_mm])
                mse_mm = float(np.mean(diff_mm ** 2))

                rel = dist_px / diag
                mre = rel * 100

                row[f"{key}_dist_px"] = dist_px
                row[f"{key}_dist_mm"] = dist_mm
                row[f"{key}_mse_px"] = mse_px
                row[f"{key}_mse_mm"] = mse_mm
                row[f"{key}_rel"] = rel
                row[f"{key}_mre"] = mre
                row[f"{key}_pred"] = (pred_x, pred_y)
                row[f"{key}_gt"] = (gt_x, gt_y)
                row[f"{key}_spacing"] = (spacing_x, spacing_y)

            # Only keep the row if at least one channel contributed.
            if row:
                records.append(row)

    df = pd.DataFrame(records)

    print(f"\n📊 PER-SLICE RESULTS ({len(df)} slices total)")
    print(f"{'':35s} {'Point1':>18s} {'Point2':>18s} {'Mean':>18s}")
    print("-" * 95)

    results = {"per_slice": df}

    metrics = [
        ("dist_px", "Euclidean Dist (px)", ".3f"),
        ("dist_mm", "Euclidean Dist (mm)", ".3f"),
        ("mse_px", "MSE (px²)", ".5f"),
        ("mse_mm", "MSE (mm²)", ".5f"),
        ("rel", "Relative Error", ".5f"),
        ("mre", "MRE (%)", ".3f"),
    ]

    for metric, label, fmt in metrics:
        p1_mean = df[f"p1_{metric}"].mean()
        p2_mean = df[f"p2_{metric}"].mean()
        p1_std = df[f"p1_{metric}"].std()
        p2_std = df[f"p2_{metric}"].std()
        mean_value = (p1_mean + p2_mean) / 2
        mean_std = (p1_std + p2_std) / 2

        print(
            f"{label:35s} "
            f"{p1_mean:{fmt}} ± {p1_std:{fmt}}   "
            f"{p2_mean:{fmt}} ± {p2_std:{fmt}}   "
            f"{mean_value:{fmt}} ± {mean_std:{fmt}}"
        )

        results[f"p1_{metric}_mean"] = p1_mean
        results[f"p1_{metric}_std"] = p1_std
        results[f"p2_{metric}_mean"] = p2_mean
        results[f"p2_{metric}_std"] = p2_std
        results[f"mean_{metric}"] = mean_value
        results[f"std_{metric}"] = mean_std

    # --- convenience aliases so existing main.py prints still work ---
    results["mean_dist"] = (results["p1_dist_px_mean"] + results["p2_dist_px_mean"]) / 2
    results["mean_mse"] = (results["p1_mse_px_mean"] + results["p2_mse_px_mean"]) / 2
    results["mean_rel"] = (results["p1_rel_mean"] + results["p2_rel_mean"]) / 2
    results["p1_dist"] = results["p1_dist_px_mean"]
    results["p2_dist"] = results["p2_dist_px_mean"]
    results["p1_mre"] = results["p1_mre_mean"]
    results["p2_mre"] = results["p2_mre_mean"]

    return results