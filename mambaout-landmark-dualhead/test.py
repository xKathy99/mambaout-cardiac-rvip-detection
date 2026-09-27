# test.py
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

from metrics import get_centers_from_heatmap


@torch.no_grad()
def test_metrics_cc(model, loader, device, image_size=(256, 256)):
    """
    Connected-component evaluation for both insertion points.
    Computes per-slice metrics, reports per-point and combined averages.
    Dataset loader is expected to yield (images, masks, spacings).
    """
    model.eval()
    H, W = image_size
    diag = (H**2 + W**2) ** 0.5
    records = []

    for batch in tqdm(loader, desc="Testing (CC method)"):
        # Unpack: images, masks, spacings
        images, targets, _ = batch # spacings not used

        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        outputs = model(images)
        B = images.shape[0]

        for i in range(B):
            row = {}
            for ch, key in enumerate(["p1", "p2"]):
                pred_heat = outputs[i, ch].cpu().numpy()
                gt_heat = targets[i, ch].cpu().numpy()

                gt_y, gt_x = ndimage.center_of_mass(gt_heat)
                pred_x, pred_y = get_centers_from_heatmap(pred_heat)

                dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                mse = float(((np.array([pred_x, pred_y]) - np.array([gt_x, gt_y])) ** 2).mean())
                rel = dist / diag
                mre = dist / diag * 100

                row[f"{key}_dist"] = dist
                row[f"{key}_mse"] = mse
                row[f"{key}_rel"] = rel
                row[f"{key}_mre"] = mre
                row[f"{key}_pred"] = (pred_x, pred_y)
                row[f"{key}_gt"] = (gt_x, gt_y)
            records.append(row)

    df = pd.DataFrame(records)
    print(f"\n📊 PER-SLICE RESULTS  ({len(df)} slices total)")
    print(f"{'':25s} {'Point1':>10s} {'Point2':>10s} {'Mean':>10s}")
    print("-" * 59)

    results = {"per_slice": df}
    for metric, label, fmt in [
        ("dist", "Euclidean Dist (px)", ".3f"),
        ("mse", "MSE", ".5f"),
        ("rel", "Relative Error", ".5f"),
        ("mre", "MRE (%)", ".3f"),
    ]:
        p1 = df[f"p1_{metric}"].mean()
        p2 = df[f"p2_{metric}"].mean()
        avg = (p1 + p2) / 2
        print(f"{label:25s} {p1:>10{fmt}} {p2:>10{fmt}} {avg:>10{fmt}}")
        results[f"p1_{metric}"] = p1
        results[f"p2_{metric}"] = p2
        results[f"mean_{metric}"] = avg

    return results