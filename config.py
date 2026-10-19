"""Configuration file for the Multimodal Product Classifier.
Defines hyperparameters, model choices, image/text preprocessing configs, and file paths.
"""

from dataclasses import dataclass
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


@dataclass
class Config:
    # --- Device Configuration ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 0  # 0 for Windows compatibility

    # --- Vision Branch ---
    # Options: 'resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v2'
    IMAGE_BACKBONE: str = "resnet18"
    IMAGE_PRETRAINED: bool = True
    FREEZE_BACKBONE_EPOCHS: int = 0  # Set > 0 to freeze CNN layers initially
    IMAGE_SIZE: int = 224
    IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGE_STD: tuple = (0.229, 0.224, 0.225)

    # --- Text Branch ---
    # Options: 'embedding_bag' (trainable embeddings with mean pooling) or 'tfidf'
    TEXT_ENCODER_TYPE: str = "embedding_bag"
    TEXT_EMBED_DIM: int = 128
    MAX_VOCAB_SIZE: int = 10000
    MIN_WORD_FREQ: int = 2
    MAX_TITLE_LEN: int = 32

    # --- Fusion & Classification ---
    DROPOUT: float = 0.3
    USE_BATCH_NORM: bool = True

    # --- Training Hyperparameters ---
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    LR_HEAD: float = 1e-3
    LR_BACKBONE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    PATIENCE: int = 3  # Early stopping patience
    SEED: int = 42

    # --- Paths ---
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    CHECKPOINT_DIR: Path = CHECKPOINT_DIR
    DATA_CSV: Path = DATA_DIR / "products.csv"
    CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best_multimodal_model.pt"
    VOCAB_PATH: Path = CHECKPOINT_DIR / "vocab.json"
    LABEL_MAP_PATH: Path = CHECKPOINT_DIR / "label_map.json"


default_config = Config()
