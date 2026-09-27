# An Automated Cardiac Landmark Detection Model
A dualhead MambaOut Encoder U-Net model for Cardiac RVIP detetion. Use case: for downstream application of AHA 17-segment regional strain map alignment 

## File Structure
```
mambaout-cardiac-rvip-detection/
├── mambaout-landmark-dualhead/
│   ├── config.py          # Paths, hyperparameters, checkpoint settings
│   ├── datasets.py        # Dataset loader, transforms
│   ├── main.py            
│   ├── metrics.py         # Evaluation metrics
│   ├── models.py          # Dual-head model: outputs RVIP and segmentation masks
│   ├── requirements       # Dependencies
│   ├── test.py            # Testing utilities
│   ├── train.py           # Training loop
│   └── utils.py           # Helper functions
│
└── mambaout-landmark-singlehead/
    ├── config.py          # Paths, hyperparameters
    ├── datasets.py        # Dataset loader, transforms
    ├── main.py            
    ├── metrics.py         # Evaluation metrics
    ├── models.py          # Single-head model: outputs RVIP only
    ├── requirements.txt   # Dependencies
    ├── test.py            # Testing utilities
    ├── train.py           # Training loop
    └── utils.py           # Helper functions
```
## Getting Started

### Dependencies
```sh
python=3.8 or above
pip
torch
torchvision
timm
nibabel
monai
numpy
scipy
pandas
matplotlib
tqdm
einops
torchsummary
```

### Installation
1. Clone the repository:
```sh
git clone https://github.com/xKathy99/mambaout-cardiac-rvip-detection.git
cd mambaout-cardiac-rvip-detection
```
2. Install dependencies for the dual-head variant:
```sh
cd mambaout-landmark-dualhead
pip install -r requirements
```

3. Install dependencies for the single-head variant:
```sh
cd mambaout-landmark-singlehead
pip install -r requirements
```

4. Modify the relevant config.py file before training:
```sh
# in config.py
IMAGEDIR        = ".../rv-points-acdc-nifti"
MASKS_DIR       = ".../rv-points-acdc-landmark"
SEG_MASKS_DIR   = ".../rv-points-acdc-segmentation" # for dual-head
TEST_IMAGEDIR   = ".../testing_nifti"
TEST_MASKS_DIR  = ".../testing_nifti_masks"
CHECKPOINT_DIR  = ".../checkpoint"  
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")
```

5. Ensure  data is arranged as expected:
```
- Images: NIfTI volumes such as .nii or .nii.gz
- Masks: 2-channel heatmap masks for the RVIP landmark data
- Dual-head only: 4-class segmentation masks for the auxiliary segmentation head
```

### Models
1. Train/test the dual-head model in main.py
```sh
cd mambaout-landmark-dualhead
python main.py
```

2. Train the single-head model
```sh
cd mambaout-landmark-singlehead
python main.py --mode train --batch_size 4 --lr 1e-4 --epochs 100 --unfreeze_epoch 100
```

3. Test the single-head model
```sh
python main.py --mode test --checkpoint ../checkpoints/best_model.pth
```