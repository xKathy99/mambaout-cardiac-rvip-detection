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
```
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
Clone the repository:
```
git clone https://github.com/xKathy99/mambaout-cardiac-rvip-detection.git
cd mambaout-cardiac-rvip-detection
```
Install dependencies for the dual-head variant:
```
cd mambaout-landmark-dualhead
pip install -r requirements
```

Install dependencies for the single-head variant:
```
cd mambaout-landmark-singlehead
pip install -r requirements
```


