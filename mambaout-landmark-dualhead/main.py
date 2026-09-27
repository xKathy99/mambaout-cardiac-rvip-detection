import torch
from torch.utils.data import DataLoader, random_split, Subset

from config import (
    IMAGEDIR, MASKS_DIR, SEG_MASKS_DIR,
    TEST_IMAGEDIR, TEST_MASKS_DIR,
    BATCH_SIZE, NUM_EPOCHS, UNFREEZE_EPOCH, LEARNING_RATE,
    SPATIAL_SIZE, CHECKPOINT_DIR, BEST_MODEL_PATH
)
from datasets import (
    Nifti2DSliceDataset, Nifti2DSliceDatasetV2,
    build_train_transforms, build_val_transforms,
    build_train_transforms_v2, build_val_transforms_v2
)
from models import UNet2D
from train import train_model, plot_history
from test import test_metrics_cc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build transforms
    train_transforms_v2 = build_train_transforms_v2(spatial_size=SPATIAL_SIZE)
    val_transforms_v2 = build_val_transforms_v2(spatial_size=SPATIAL_SIZE)

    # Full dataset for split indices
    full_dataset = Nifti2DSliceDatasetV2(
        image_dir=IMAGEDIR,
        mask_dir=MASKS_DIR,
        seg_dir=SEG_MASKS_DIR,
        transform=None,
        slice_axis=2
    )
    total_len = len(full_dataset)
    train_len = int(0.75 * total_len)
    val_len = total_len - train_len

    train_indices, val_indices = random_split(
        range(total_len),
        [train_len, val_len],
        generator=torch.Generator().manual_seed(32)
    )

    # Datasets with transforms
    train_dataset = Nifti2DSliceDatasetV2(
        image_dir=IMAGEDIR, mask_dir=MASKS_DIR, seg_dir=SEG_MASKS_DIR,
        transform=train_transforms_v2, slice_axis=2
    )
    val_dataset = Nifti2DSliceDatasetV2(
        image_dir=IMAGEDIR, mask_dir=MASKS_DIR, seg_dir=SEG_MASKS_DIR,
        transform=val_transforms_v2, slice_axis=2
    )

    train_dataset = Subset(train_dataset, list(train_indices))
    val_dataset = Subset(val_dataset, list(val_indices))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Total slices : {total_len}")
    print(f"Train slices : {train_len}")
    print(f"Val slices   : {val_len}")

    # Model
    model = UNet2D(in_ch=1, num_classes=2, num_seg_classes=4)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=NUM_EPOCHS,
        unfreeze_epoch=UNFREEZE_EPOCH,
        use_amp=True,
        ckpt_dir=CHECKPOINT_DIR,
        best_path=BEST_MODEL_PATH
    )
    plot_history(history)

    # Load best model
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded best model from {BEST_MODEL_PATH}")

    # Test on test set (2-output dataset)
    val_transforms = build_val_transforms(spatial_size=SPATIAL_SIZE)
    test_dataset = Nifti2DSliceDataset(
        image_dir=TEST_IMAGEDIR,
        mask_dir=TEST_MASKS_DIR,
        transform=val_transforms,
        slice_axis=2,
        spatial_size=SPATIAL_SIZE,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test dataset size: {len(test_dataset)} slices")

    results = test_metrics_cc(model, test_loader, device, image_size=SPATIAL_SIZE)
    print(f'results["mean_dist"] {results["mean_dist"]}')
    print(f'results["mean_mse"] {results["mean_mse"]}')
    print(f'results["mean_rel"] {results["mean_rel"]}')
    print(f'results["p1_dist"] {results["p1_dist"]}')
    print(f'results["p2_dist"] {results["p2_dist"]}')
    print(f'results["p1_mre"] {results["p1_mre"]}')
    print(f'results["p2_mre"] {results["p2_mre"]}')


if __name__ == "__main__":
    main()