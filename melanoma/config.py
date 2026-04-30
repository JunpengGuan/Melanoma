from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent

# Default layout follows data/ISIC2017
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "ISIC2017" / "Train" / "ISIC-2017_Training_Data"
DEFAULT_LABEL_CSV = PROJECT_ROOT / "data" / "ISIC2017" / "Train" / "ISIC-2017_Training_Part3_GroundTruth.csv"

# ISIC2017 test release
DEFAULT_TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "ISIC2017" / "Test" / "ISIC-2017_Test_v2_Data"
DEFAULT_TEST_LABEL_CSV = PROJECT_ROOT / "data" / "ISIC2017" / "Test" / "ISIC-2017_Test_v2_Part3_GroundTruth.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_last.pt"

DEFAULT_IMG_SIZE = 224

# Segmentation GT
DEFAULT_SEG_MASK_DIR = PROJECT_ROOT / "data" / "ISIC2017" / "Train" / "ISIC-2017_Training_Part1_GroundTruth"

SEG_IMG_SIZE = 256
DEFAULT_UNET_CKPT = PROJECT_ROOT / "checkpoints" / "unet_last.pt"
DEFAULT_METHOD2_LR = PROJECT_ROOT / "checkpoints" / "method2_lr.joblib"
DEFAULT_METHOD2_XGB = PROJECT_ROOT / "checkpoints" / "method2_xgb.json"

METHOD1_CONFIG_YAML = PACKAGE_ROOT / "method1" / "config.yaml"
METHOD2_CONFIG_YAML = PACKAGE_ROOT / "method2" / "config.yaml"
