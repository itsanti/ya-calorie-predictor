import torch
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent

class Config:
    SEED = 42

    TEXT_MODEL_UNFREEZE = "transformer.layer.5"
    IMAGE_MODEL_UNFREEZE = "stages.3"
    
    BATCH_SIZE = 64
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-5
    REGRESSOR_LR = 1e-3

    EPOCHS = 15
    DROPOUT = 0.15
    HIDDEN_DIM = 256

    SAVE_PATH = PROJECT_PATH / "model" / "best_model.pth"
    TRAIN_PATH = PROJECT_PATH / "data" / "train.csv"
    VAL_PATH = PROJECT_PATH / "data" / "val.csv"
    TEST_PATH = PROJECT_PATH / "data" / "test.csv"
    IMAGES_PATH = PROJECT_PATH / "data" / "images"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    USE_AMP = True
    GRAD_CLIP = 1.0
    PATIENCE = 3
