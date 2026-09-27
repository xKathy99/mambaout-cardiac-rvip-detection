# config.py
import os

IMAGEDIR = '.../rv-points-acdc-nifti-images'
MASKS_DIR = '.../rv-points-acdc-nifti-cleaned-heatmaps-2ch'
SEG_MASKS_DIR = '.../rv-points-acdc-nifti-seg-masks'

TEST_IMAGEDIR = '.../testing_nifti_2'
TEST_MASKS_DIR = '.../testing_nifti_masks_class-2ch-heatmaps'

# Training hyperparams
BATCH_SIZE = 4
NUM_EPOCHS = 100
UNFREEZE_EPOCH = 100
LEARNING_RATE = 1e-4
SEG_WEIGHT = 0.3
SPATIAL_SIZE = (256, 256)

# Checkpoint paths
CHECKPOINT_DIR = '.../20260618-v6-dualhead-mambavar'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, '20260618-v6-dualhead-mambavar-best.pth')