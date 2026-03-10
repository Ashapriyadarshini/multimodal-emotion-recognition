import torch

# Paths
DATA_PATH = "data/RAVDESS"
METADATA_FILE = "../data/ravdess_metadata.csv"

# Training settings
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 0.0001

# Model settings
TEXT_DIM = 768
AUDIO_DIM = 128
VISUAL_DIM = 512
FUSION_DIM = 256
NUM_CLASSES = 8

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"