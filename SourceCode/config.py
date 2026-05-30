# RAF-DB Dataset Configuration
TRAIN_DIR = "dataset/DATASET/train"
TEST_DIR = "dataset/DATASET/test"

# Model Configuration
MODEL_NAME = "resnet50"
NUM_CLASSES = 7
PRETRAINED = True

# Image Configuration
IMAGE_SIZE = 224
RAF_DB_MEAN = [0.57535914, 0.44928582, 0.40079932]
RAF_DB_STD = [0.20735591, 0.18981615, 0.18132027]

# Training Configuration
BATCH_SIZE = 64
NUM_WORKERS = 2
VAL_SPLIT = 0.2
EPOCHS = 100
DEVICE = "cuda"
SEED = 42

# Optimization Configuration
LR = 0.01
LR_SCHEDULER = "step"
LR_FACTOR = 0.1
LR_PATIENCE = 30
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Regularization & Dropout
USE_DROPOUT = True
DROPOUT_PROB = 0.3
USE_REGULARIZATION = True
REGULARIZATION_TYPE = "l2"
REGULARIZATION_FACTOR = 5e-4

# Checkpoint & Logging
CHECKPOINT_DIR = "checkpoint"
LOG_DIR = "log"
PRINT_FREQ = 10
SAVE_FREQ = 5

# Inference Configuration
CONFIDENCE_THRESHOLD = 0.5

# RAF-DB Emotion Labels (0-indexed)
EMOTION_LABELS = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}