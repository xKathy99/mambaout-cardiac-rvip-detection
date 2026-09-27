import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from metrics import compute_distance


# Losses
heatmap_criterion = nn.MSELoss()
seg_criterion = nn.CrossEntropyLoss()
SEG_WEIGHT = 0.3


def set_encoder_trainable(model, trainable: bool):
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def get_encoder_params(model):
    return list(model.encoder.parameters())


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_dist = 0.0

    for images, masks, segs in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        segs = segs.to(device, dtype=torch.float32)

        seg_targets = segs.squeeze(1).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(scaler is not None)):
            heatmap, seg_logits = model(images)
            loss_heatmap = heatmap_criterion(heatmap, masks)
            loss_seg = seg_criterion(seg_logits, seg_targets)
            loss = loss_heatmap + SEG_WEIGHT * loss_seg

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dist += compute_distance(heatmap, masks)

    return total_loss / len(loader), total_dist / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dist = 0.0

    for images, masks, segs in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        heatmap = model(images)
        total_loss += heatmap_criterion(heatmap, masks).item()
        total_dist += compute_distance(heatmap, masks)

    return total_loss / len(loader), total_dist / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=100,
    unfreeze_epoch=50,
    use_amp=True,
    ckpt_dir="checkpoints",
    best_path="best_model.pth",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)

    scaler = GradScaler("cuda") if use_amp else None
    set_encoder_trainable(model, False)

    history = {"train_loss": [], "val_loss": [], "train_dist": [], "val_dist": []}
    best_dist = float("inf")
    encoder_unfrozen = False

    try:
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")

            # if (not encoder_unfrozen) and (epoch >= unfreeze_epoch):
                # print("🔓 Unfreezing encoder...")
                # set_encoder_trainable(model, True)
                # encoder_unfrozen = True

                # decoder_lr = optimizer.param_groups[0]["lr"]
                # encoder_lr = decoder_lr * 0.1
                # encoder_ids = {id(p) for p in get_encoder_params(model)}

                # optimizer = torch.optim.AdamW([
                    # {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": decoder_lr},
                    # {"params": get_encoder_params(model), "lr": encoder_lr},
                # ])
                # if use_amp:
                    # scaler = GradScaler("cuda")
                # print(f"  Decoder lr: {decoder_lr:.2e} | Encoder lr: {encoder_lr:.2e}")

            train_loss, train_dist = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_loss, val_dist = evaluate(model, val_loader, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_dist"].append(train_dist)
            history["val_dist"].append(val_dist)

            print(f"Train Loss: {train_loss:.5f} | Train Dist: {train_dist:.3f}px")
            print(f"Val Loss:   {val_loss:.5f} | Val Dist:   {val_dist:.3f}px")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss, "val_loss": val_loss,
                "train_dist": train_dist, "val_dist": val_dist,
                "encoder_unfrozen": encoder_unfrozen,
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth"))

            if val_dist < best_dist:
                best_dist = val_dist
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dist": best_dist,
                }, best_path)
                print(f"✓ Best model updated (val_dist={best_dist:.3f}px)")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving progress...")
    except Exception as e:
        print(f"\n❌ Training crashed at epoch {epoch+1}: {type(e).__name__}: {e}")
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