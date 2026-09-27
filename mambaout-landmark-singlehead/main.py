# main.py
import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
import argparse

from config import *
from datasets import Nifti2DSliceDataset, build_train_transforms, build_val_transforms
from models import UNet2D
from train import train_model, plot_history
from test import load_model, test_metrics_cc
from utils import plot_predictions_multi_centers


def main():
    parser = argparse.ArgumentParser(description="MambaOut UNet for RV Landmark Detection")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--unfreeze_epoch", type=int, default=UNFREEZE_EPOCH)
    parser.add_argument("--checkpoint", type=str, default=BEST_PATH_WEIGHT, help="Path to checkpoint for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build transforms
    train_transforms = build_train_transforms(spatial_size=SPATIAL_SIZE)
    val_transforms = build_val_transforms(spatial_size=SPATIAL_SIZE)

    # Build datasets
    full_dataset = Nifti2DSliceDataset(
        image_dir=IMAGEDIR,
        mask_dir=MASKS_DIR,
        transform=None,
        slice_axis=2,
        spatial_size=SPATIAL_SIZE
    )
    total_len = len(full_dataset)
    train_len = int(0.75 * total_len)
    val_len = total_len - train_len

    train_indices, val_indices = random_split(
        range(total_len),
        [train_len, val_len],
        generator=torch.Generator().manual_seed(32)
    )

    train_dataset = Nifti2DSliceDataset(
        image_dir=IMAGEDIR, mask_dir=MASKS_DIR,
        transform=train_transforms, slice_axis=2,
        spatial_size=SPATIAL_SIZE
    )
    val_dataset = Nifti2DSliceDataset(
        image_dir=IMAGEDIR, mask_dir=MASKS_DIR,
        transform=val_transforms, slice_axis=2,
        spatial_size=SPATIAL_SIZE
    )

    train_dataset = Subset(train_dataset, list(train_indices))
    val_dataset = Subset(val_dataset, list(val_indices))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'Total slices : {total_len}')
    print(f'Train slices : {train_len}')
    print(f'Val slices   : {val_len}')

    # Model
    model = UNet2D(in_ch=IN_CH, num_classes=NUM_CLASSES)
    model = model.to(device)

    if args.mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            unfreeze_epoch=args.unfreeze_epoch,
            use_amp=True,
            ckpt_dir=CKPT_DIR,
            best_path=args.checkpoint
        )
        plot_history(history)

    elif args.mode == "test":
        # Load best model
        model = load_model(model, args.checkpoint, device)
        print(f"✅ Loaded best model from {args.checkpoint}")

        # Test dataset
        test_dataset = Nifti2DSliceDataset(
            image_dir=TEST_IMAGEDIR,
            mask_dir=TEST_MASKS_DIR,
            transform=val_transforms,
            slice_axis=2,
            spatial_size=SPATIAL_SIZE
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Test dataset size: {len(test_dataset)} slices")

        results = test_metrics_cc(model, test_loader, device)
        print(f'results["mean_dist"] {results["mean_dist"]}')
        print(f'results["mean_mse"] {results["mean_mse"]}')
        print(f'results["mean_rel"] {results["mean_rel"]}')
        print(f'results["p1_dist"] {results["p1_dist"]}')
        print(f'results["p2_dist"] {results["p2_dist"]}')
        print(f'results["p1_mre"] {results["p1_mre"]}')
        print(f'results["p2_mre"] {results["p2_mre"]}')

        # Plot predictions
        plot_predictions_multi_centers(model, test_loader, device)


if __name__ == "__main__":
    main()