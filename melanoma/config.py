from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default layout: images + CSV live next to this repo (see FinalProject/)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "ISBI2016_ISIC_Part1_Training_Data"
DEFAULT_LABEL_CSV = PROJECT_ROOT / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"

# ISBI2016 Part 3B test release (379 images; CSV uses 0.0/1.0 labels)
DEFAULT_TEST_IMAGE_DIR = PROJECT_ROOT / "ISBI2016_ISIC_Part3B_Test_Data"
DEFAULT_TEST_LABEL_CSV = PROJECT_ROOT / "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_last.pt"

DEFAULT_IMG_SIZE = 224

# Segmentation GT (nested folder in ISBI2016 release)
DEFAULT_SEG_MASK_DIR = (
    PROJECT_ROOT
    / "ISBI2016_ISIC_Part1_Training_GroundTruth"
    / "ISBI2016_ISIC_Part1_Training_GroundTruth"
)

SEG_IMG_SIZE = 256
DEFAULT_UNET_CKPT = PROJECT_ROOT / "checkpoints" / "unet_last.pt"
DEFAULT_METHOD2_LR = PROJECT_ROOT / "checkpoints" / "method2_lr.joblib"
DEFAULT_METHOD2_XGB = PROJECT_ROOT / "checkpoints" / "method2_xgb.json"
