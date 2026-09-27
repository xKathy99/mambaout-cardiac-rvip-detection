import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


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