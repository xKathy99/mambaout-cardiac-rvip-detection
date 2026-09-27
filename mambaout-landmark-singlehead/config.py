import os

# Paths
IMAGEDIR = '.../rv-points-acdc-nifti-images'
MASKS_DIR = '.../rv-points-acdc-nifti-cleaned-heatmaps-2ch'
TEST_IMAGEDIR = '.../testing_nifti_2'
TEST_MASKS_DIR = '.../testing_nifti_masks_class-2ch-heatmaps'

CKPT_DIR = "checkpoints"
BEST_PATH_WEIGHT = os.path.join(CKPT_DIR, "best_model.pth")

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 100
UNFREEZE_EPOCH = 100
SPATIAL_SIZE = (256, 256)
NUM_CLASSES = 2
IN_CH = 1