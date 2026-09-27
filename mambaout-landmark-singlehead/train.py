# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from metrics import get_centers_from_heatmap

# Loss
heatmap_criterion = nn.MSELoss()


def _batch_center_distance(outputs, targets, skip_empty_gt=True):
    """
    Mean Euclidean distance (px) using the SAME numpy-based center
    extraction (`get_centers_from_heatmap`) as `test_metrics_cc`.

    Averages over all valid (batch, point) pairs. Empty GT heatmaps are
    skipped so they don't contribute a meaningless "image center" target.
    """
    pred_np = outputs.detach().float().cpu().numpy()
    gt_np = targets.detach().float().cpu().numpy()

    B, C = pred_np.shape[:2]
    total = 0.0
    count = 0

    for b in range(B):
        for c in range(C):
            gt_heat = gt_np[b, c]
            if skip_empty_gt and gt_heat.max() <= 0:
                continue
            gx, gy = get_centers_from_heatmap(gt_heat)
            px, py = get_centers_from_heatmap(pred_np[b, c])
            total += float(np.hypot(px - gx, py - gy))
            count += 1

    return total / count if count > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_dist = 0.0

    for images, masks, _ in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(scaler is not None)):
            heatmap = model(images)
            loss = heatmap_criterion(heatmap, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dist += _batch_center_distance(heatmap, masks)

    return total_loss / len(loader), total_dist / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dist = 0.0

    for images, masks, _ in tqdm(loader, desc='Val', leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        heatmap = model(images)

        total_loss += heatmap_criterion(heatmap, masks).item()
        total_dist += _batch_center_distance(heatmap, masks)

    return total_loss / len(loader), total_dist / len(loader)


def set_encoder_trainable(model, trainable: bool):
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def get_encoder_params(model):
    return list(model.encoder.parameters())


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=100,
    unfreeze_epoch=100,
    use_amp=True,
    ckpt_dir='checkpoints',
    best_path='best_model.pth',
):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)

    # Only enable AMP on CUDA; otherwise silently disable it.
    use_amp = use_amp and device.type == 'cuda'
    scaler = GradScaler(device.type) if use_amp else None

    set_encoder_trainable(model, False)

    history = {'train_loss': [], 'val_loss': [], 'train_dist': [], 'val_dist': []}
    best_dist = float('inf')
    encoder_unfrozen = False

    try:
        for epoch in range(epochs):
            print(f'\nEpoch [{epoch+1}/{epochs}]')

            # --- optional encoder unfreeze (kept commented as in your original) ---
            # if (not encoder_unfrozen) and (epoch >= unfreeze_epoch):
            #     print('🔓 Unfreezing encoder...')
            #     set_encoder_trainable(model, True)
            #     encoder_unfrozen = True
            #
            #     decoder_lr = optimizer.param_groups[0]['lr']
            #     encoder_lr = decoder_lr * 0.1
            #     encoder_ids = {id(p) for p in get_encoder_params(model)}
            #
            #     optimizer = torch.optim.AdamW([
            #         {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': decoder_lr},
            #         {'params': get_encoder_params(model), 'lr': encoder_lr},
            #     ])
            #
            #     if use_amp:
            #         scaler = GradScaler(device.type)
            #
            #     print(f'  Decoder lr: {decoder_lr:.2e} | Encoder lr: {encoder_lr:.2e}')

            train_loss, train_dist = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_loss, val_dist = evaluate(model, val_loader, device)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dist'].append(train_dist)
            history['val_dist'].append(val_dist)

            print(f'Train Loss: {train_loss:.5f} | Train Dist: {train_dist:.3f}px')
            print(f'Val Loss:   {val_loss:.5f} | Val Dist:   {val_dist:.3f}px')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_dist': train_dist, 'val_dist': val_dist,
                'encoder_unfrozen': encoder_unfrozen,
            }, os.path.join(ckpt_dir, f'epoch_{epoch+1:03d}.pth'))

            if val_dist < best_dist:
                best_dist = val_dist
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_dist': best_dist,
                }, best_path)
                print(f'✓ Best model updated (val_dist={best_dist:.3f}px)')

    except KeyboardInterrupt:
        print('\n⚠️ Training interrupted. Saving progress...')
    except Exception as e:
        print(f'\n❌ Training crashed at epoch {epoch+1}: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
    finally:
        return history


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Train vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, history["train_dist"], label="Train Distance")
    axes[1].plot(epochs, history["val_dist"], label="Val Distance")
    axes[1].set_title("RV Insertion Point Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Euclidean Distance (pixels)")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()