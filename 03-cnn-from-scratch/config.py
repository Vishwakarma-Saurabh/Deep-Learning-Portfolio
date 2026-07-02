"""Central configuration for the CNN-from-scratch project."""

# Data
DATA_DIR = './mnist_data'
NUM_CLASSES = 10
INPUT_CHANNELS = 1
IMAGE_SIZE = 28

# Training
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'  # 'adam' or 'sgd'
MOMENTUM = 0.9  # used only for SGD

# Model
CONV1_FILTERS = 8
CONV2_FILTERS = 16
DENSE_UNITS = 64
DROPOUT_RATE = 0.5

# Augmentation
USE_AUGMENTATION = True

# Misc
RANDOM_SEED = 42
CHECKPOINT_PATH = './checkpoints/cnn_model.pkl'
PLOTS_DIR = './plots'