# metrics.py
import numpy as np
import torch
from scipy import ndimage


def soft_argmax(heatmap, beta=100):
    """heatmap: [B, 2, H, W] -> [B, 2, 2] (x, y)"""
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    prob = torch.softmax(flat * beta, dim=2)
    xs = torch.arange(W, device=heatmap.device).repeat(H).float()
    ys = torch.arange(H, device=heatmap.device).repeat_interleave(W).float()
    x = torch.sum(prob * xs, dim=2)
    y = torch.sum(prob * ys, dim=2)
    return torch.stack([x, y], dim=2)


@torch.no_grad()
def compute_distance(outputs, targets):
    """Mean Euclidean distance (px) averaged over both points and batch."""
    pred_coords = soft_argmax(outputs.detach())
    gt_coords = soft_argmax(targets)
    dist = torch.norm(pred_coords - gt_coords, dim=2)
    return dist.mean().item()


def hard_argmax(heatmap):
    """heatmap: [B, 2, H, W] -> [B, 2, 2] (x, y)"""
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    idx = torch.argmax(flat, dim=2)
    y = idx // W
    x = idx % W
    return torch.stack([x, y], dim=2).float()


def get_centers_from_heatmap(heatmap):
    """
    Extract peak center from a single 2D heatmap (H, W) using
    connected components. Returns one (x, y) — the component with
    the highest peak value.
    """
    threshold = heatmap.mean() + heatmap.std()
    binary = heatmap > threshold
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (float(x), float(y))

    best = max(
        range(1, num_features + 1),
        key=lambda k: heatmap[labeled == k].max()
    )
    cy, cx = ndimage.center_of_mass(heatmap, labeled, best)
    return (float(cx), float(cy))