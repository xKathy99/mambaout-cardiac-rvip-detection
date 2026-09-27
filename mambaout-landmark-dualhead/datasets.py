import os
from dataclasses import dataclass
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from torch.utils.data import Dataset


def build_train_transforms(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="bilinear"),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.3),
            ToTensord(keys=["image", "mask"]),
        ]
    )


def build_val_transforms(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="bilinear"),
            ToTensord(keys=["image", "mask"]),
        ]
    )


def build_train_transforms_v2(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="bilinear"),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest"),
            RandFlipd(keys=["image", "mask", "seg"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask", "seg"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.3),
            ToTensord(keys=["image", "mask", "seg"]),
        ]
    )


def build_val_transforms_v2(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="bilinear"),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest"),
            ToTensord(keys=["image", "mask", "seg"]),
        ]
    )


@dataclass(frozen=True)
class SliceIndex:
    image_file: str
    mask_file: str
    slice_idx: int


class Nifti2DSliceDataset(Dataset):
    """
    Build a 2D slice dataset from paired NIfTI volumes.

    Returns
    -------
    image  : [1,256,256]
    mask   : [2,256,256]
    spacing: [x_mm_per_pixel, y_mm_per_pixel]
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Compose = None,
        slice_axis: int = 2,
        drop_empty_mask_slices: bool = False,
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.slice_axis = slice_axis
        self.drop_empty_mask_slices = drop_empty_mask_slices
        self.spatial_size = spatial_size

        self.pairs = self._build_pairs()
        self.slice_map = self._build_slice_map()

        if not self.slice_map:
            raise RuntimeError("No samples found. Check paths and filenames.")

    # -------------------------------------------------------
    # Find image-mask pairs
    # -------------------------------------------------------

    def _build_pairs(self):
        image_files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")
        )
        pairs = []
        for image_name in image_files:
            mask_path = os.path.join(self.mask_dir, image_name)
            if os.path.exists(mask_path):
                pairs.append((image_name, image_name))
        if not pairs:
            raise RuntimeError("No matching NIfTI image/mask pairs found")
        return pairs

    # -------------------------------------------------------
    # Build slice index
    # -------------------------------------------------------

    def _build_slice_map(self):
        slice_map = []
        for image_file, mask_file in self.pairs:
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image_img = nib.load(image_path)
            mask_img = nib.load(mask_path)

            img_shape = image_img.shape
            mask_shape = mask_img.shape

            if img_shape != mask_shape[:3]:
                raise ValueError(f"Shape mismatch {img_shape} vs {mask_shape}")

            if mask_shape[3] != 2:
                raise ValueError("Mask must have 2 channels")

            num_slices = img_shape[self.slice_axis]

            for idx in range(num_slices):
                if self.drop_empty_mask_slices:
                    mask_data = np.asanyarray(mask_img.dataobj)
                    mask_slice = np.take(mask_data, idx, axis=self.slice_axis)
                    if mask_slice.max() == 0:
                        continue
                slice_map.append(SliceIndex(image_file, mask_file, idx))
        return slice_map

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        item = self.slice_map[idx]

        image_path = os.path.join(self.image_dir, item.image_file)
        mask_path = os.path.join(self.mask_dir, item.mask_file)

        # -------------------------
        # Load NIfTI
        # -------------------------
        image_img = nib.load(image_path)
        image_data = np.asanyarray(image_img.dataobj, dtype=np.float32)
        mask_data = np.asanyarray(
            nib.load(mask_path).dataobj, dtype=np.float32
        )

        # -------------------------
        # Original spacing
        # -------------------------
        voxel_spacing = image_img.header.get_zooms()
        spacing_x = voxel_spacing[0]
        spacing_y = voxel_spacing[1]

        # -------------------------
        # Extract slice
        # -------------------------
        image_2d = np.take(image_data, item.slice_idx, axis=self.slice_axis)
        mask_2d = np.take(mask_data, item.slice_idx, axis=self.slice_axis)

        original_H, original_W = image_2d.shape

        # -------------------------
        # Convert channels
        # -------------------------
        image_2d = np.expand_dims(image_2d, axis=0)
        mask_2d = np.moveaxis(mask_2d, -1, 0).copy()

        sample = {"image": image_2d, "mask": mask_2d}

        # -------------------------
        # Apply MONAI transforms
        # -------------------------
        if self.transform:
            sample = self.transform(sample)

        # -------------------------
        # Update spacing after resize
        # -------------------------
        new_H, new_W = self.spatial_size
        resized_spacing_x = spacing_x * (original_W / new_W)
        resized_spacing_y = spacing_y * (original_H / new_H)

        spacing = torch.tensor(
            [resized_spacing_x, resized_spacing_y],
            dtype=torch.float32,
        )

        return sample["image"], sample["mask"], spacing


@dataclass(frozen=True)
class SliceIndexV2:
    image_file: str
    mask_file: str
    seg_file: str
    slice_idx: int


class Nifti2DSliceDatasetV2(Dataset): # returns image, mask and segmentation mask (in that order)
    """
    3-output dataset: returns (image, mask, seg).
    Image: (H, W, D), Mask: (H, W, D, 2), Seg: (H, W, D)
    Output: image [1, H, W], mask [2, H, W], seg [1, H, W]
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        seg_dir: str,
        transform: Compose = None,
        slice_axis: int = 2,
        drop_empty_mask_slices: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.slice_axis = slice_axis
        self.drop_empty_mask_slices = drop_empty_mask_slices

        self.pairs = self._build_pairs()
        self.slice_map = self._build_slice_map()

        if not self.slice_map:
            raise RuntimeError("No training samples were built. Check paths and naming.")

    def _build_pairs(self) -> List[Tuple[str, str, str]]:
        for d in [self.image_dir, self.mask_dir, self.seg_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")
        image_files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")
        )
        pairs = []
        for image_name in image_files:
            mask_path = os.path.join(self.mask_dir, image_name)
            seg_path = os.path.join(self.seg_dir, image_name)
            if os.path.exists(mask_path) and os.path.exists(seg_path):
                pairs.append((image_name, image_name, image_name))
        if not pairs:
            raise RuntimeError("No matched image/mask/seg triplets found.")
        return pairs

    def _build_slice_map(self) -> List[SliceIndexV2]:
        slice_map = []
        for image_file, mask_file, seg_file in self.pairs:
            image_img = nib.load(os.path.join(self.image_dir, image_file))
            mask_img = nib.load(os.path.join(self.mask_dir, mask_file))
            seg_img = nib.load(os.path.join(self.seg_dir, seg_file))

            img_shape = image_img.shape
            mask_shape = mask_img.shape
            seg_shape = seg_img.shape

            if img_shape != mask_shape[:3]:
                raise ValueError(f"Shape mismatch (image vs mask): {img_shape} vs {mask_shape[:3]}")
            if img_shape != seg_shape:
                raise ValueError(f"Shape mismatch (image vs seg): {img_shape} vs {seg_shape}")
            if len(mask_shape) != 4 or mask_shape[3] != 2:
                raise ValueError(f"Expected 2-channel mask for {mask_file}, got {mask_shape}")

            num_slices = img_shape[self.slice_axis]
            for slice_idx in range(num_slices):
                if self.drop_empty_mask_slices:
                    mask_data = np.asanyarray(mask_img.dataobj)
                    mask_2d = np.take(mask_data, slice_idx, axis=self.slice_axis)
                    if mask_2d.max() == 0:
                        continue
                slice_map.append(SliceIndexV2(image_file, mask_file, seg_file, slice_idx))
        return slice_map

    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int):
        s = self.slice_map[idx]
        image_data = np.asanyarray(nib.load(os.path.join(self.image_dir, s.image_file)).dataobj, dtype=np.float32)
        mask_data = np.asanyarray(nib.load(os.path.join(self.mask_dir, s.mask_file)).dataobj, dtype=np.float32)
        seg_data = np.asanyarray(nib.load(os.path.join(self.seg_dir, s.seg_file)).dataobj, dtype=np.float32)

        image_2d = np.take(image_data, s.slice_idx, axis=self.slice_axis)
        mask_2d = np.take(mask_data, s.slice_idx, axis=self.slice_axis)
        seg_2d = np.take(seg_data, s.slice_idx, axis=self.slice_axis)

        image_2d = np.expand_dims(image_2d, axis=0)
        mask_2d = np.moveaxis(mask_2d, -1, 0).copy()
        seg_2d = np.expand_dims(seg_2d, axis=0)

        sample = {"image": image_2d, "mask": mask_2d, "seg": seg_2d}
        if self.transform:
            sample = self.transform(sample)
        return sample["image"], sample["mask"], sample["seg"] # returns image, mask and segmentation mask (in that order)