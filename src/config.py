# src/config.py

from pathlib import Path

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_DIR = RAW_DATA_DIR / "train"

TEST_DIR = RAW_DATA_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# Image Settings


IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 1

# Dataset Settings

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

# Training

EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_CLASSES = 7