# utils.py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

from metrics import get_centers_from_heatmap


def plot_nifti_with_mask(image_path, mask_path, colormap='Blues', alpha=0.5, point_mask=False):
    """
    Plot all slices of a NIfTI volume with a mask superimposed.
    """
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    if img.shape != mask.shape:
        print("Image and mask shapes do not match!")
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
        return

    num_slices = img.shape[2]

    for slice_idx in range(num_slices):
        img_slice = img[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice, cmap='gray')

        if point_mask:
            y_coords, x_coords = np.where(mask_slice > 0)
            if len(x_coords) > 0:
                plt.scatter(x_coords, y_coords, c='green', s=50)
        else:
            plt.imshow(mask_slice, cmap=colormap, alpha=alpha)

        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
        plt.show()


@torch.no_grad()
def plot_predictions_multi_centers(
    model,
    loader,
    device,
    batch_idx=0,
    num_samples=4,
    alpha=0.5
):
    """
    Visualize GT and Pred heatmaps with centers for both points.
    """
    model.eval()

    images = None
    targets = None
    spacings = None

    for i, (img, tgt, spacing) in enumerate(loader):
        if i == batch_idx:
            images = img
            targets = tgt
            spacings = spacing
            break

    if images is None:
        raise ValueError(f"batch_idx {batch_idx} out of range")

    images_device = images.to(device, dtype=torch.float32)
    targets_device = targets.to(device, dtype=torch.float32)
    outputs = model(images_device)

    images_np = images.cpu().numpy()
    targets_np = targets_device.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    num_samples = min(num_samples, images_np.shape[0])
    POINT_LABELS = ["Point 1", "Point 2"]
    all_centers = []

    for i in range(num_samples):
        img = images_np[i, 0]
        spacing_x = spacings[i][0].item()
        spacing_y = spacings[i][1].item()

        print(f"\nSample {i}")
        print(f"Pixel spacing: {spacing_x:.4f} x {spacing_y:.4f} mm/pixel")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Sample {i}", fontsize=13, fontweight="bold")

        axes[0, 0].set_title("Point 1 GT")
        axes[0, 1].set_title("Point 1 Prediction")
        axes[1, 0].set_title("Point 2 GT")
        axes[1, 1].set_title("Point 2 Prediction")

        sample_centers = {}

        for ch, point_label in enumerate(POINT_LABELS):
            gt_heat = targets_np[i, ch]
            pred_heat = outputs_np[i, ch]

            gt_center = get_centers_from_heatmap(gt_heat)
            pred_center = get_centers_from_heatmap(pred_heat)

            gt_x, gt_y = gt_center
            pred_x, pred_y = pred_center

            dx_px = pred_x - gt_x
            dy_px = pred_y - gt_y
            dist_px = np.sqrt(dx_px**2 + dy_px**2)

            dx_mm = dx_px * spacing_x
            dy_mm = dy_px * spacing_y
            dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)

            print(f"\n{point_label}")
            print(f"  GT center   : ({gt_x:.2f}, {gt_y:.2f})")
            print(f"  Pred center : ({pred_x:.2f}, {pred_y:.2f})")
            print(f"  Error       : {dist_px:.3f} px")
            print(f"  Error       : {dist_mm:.3f} mm")

            sample_centers[point_label] = {
                "gt": gt_center,
                "pred": pred_center,
                "error_px": dist_px,
                "error_mm": dist_mm
            }

            axes[ch, 0].imshow(img, cmap="gray")
            axes[ch, 0].imshow(gt_heat, cmap="jet", alpha=alpha)
            axes[ch, 0].scatter(gt_x, gt_y, c="cyan", s=120, marker="x", linewidths=2, label="GT")
            axes[ch, 0].scatter(pred_x, pred_y, c="red", s=120, marker="x", linewidths=2, label="Pred")
            axes[ch, 0].set_ylabel(point_label, fontsize=11, fontweight="bold")
            axes[ch, 0].axis("off")

            axes[ch, 1].imshow(img, cmap="gray")
            axes[ch, 1].imshow(pred_heat, cmap="jet", alpha=alpha)
            axes[ch, 1].scatter(pred_x, pred_y, c="red", s=120, marker="x", linewidths=2)
            axes[ch, 1].scatter(gt_x, gt_y, c="cyan", s=120, marker="x", linewidths=2)
            axes[ch, 1].set_title(f"{point_label} Prediction\nError={dist_mm:.2f} mm")
            axes[ch, 1].axis("off")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="x", color="cyan", markersize=10, linewidth=0, label="GT center"),
            Line2D([0], [0], marker="x", color="red", markersize=10, linewidth=0, label="Pred center")
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        plt.show()

        all_centers.append(sample_centers)

    return all_centers