import numpy as np

class Config:
    
    # ========== Random Seed ==========
    RANDOM_SEED = 42
    
    # ========== Data ==========
    INPUT_SHAPE = (1, 28, 28)  # (channels, height, width)
    NUM_CLASSES = 10
    
    # ========== CNN Architecture ==========
    # Conv Layers: (in_channels, out_channels, kernel_size, padding, stride)
    CONV_LAYERS = [
        {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'stride': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1},
    ]
    
    # Pooling Layers: (pool_size, stride)
    POOL_LAYERS = [
        {'pool_size': 2, 'stride': 2},  # After first conv
        {'pool_size': 2, 'stride': 2},  # After second conv
    ]
    
    # Fully Connected Layers
    FC_LAYERS = [128, 10]  # Hidden units, then output
    
    # ========== Training ==========
    EPOCHS = 3
    BATCH_SIZE = 64
    MAX_TRAIN_SAMPLES = 4000
    MAX_VAL_SAMPLES = 1000
    MAX_TEST_SAMPLES = 1000
    # ========== Optimization ==========
    OPTIMIZER = 'adam'  # 'sgd', 'momentum', 'adam', 'adamw'
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    
    # Adam specific
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    
    # ========== Regularization ==========
    DROPOUT_RATE = 0.5  # For FC layers
    USE_BATCH_NORM = True
    USE_AUGMENTATION = False
    LABEL_SMOOTHING = 0.1
    
    # ========== Learning Rate Scheduling ==========
    LR_SCHEDULE = 'cosine'  # 'step', 'cosine', 'warmup_cosine'
    WARMUP_EPOCHS = 5
    
    # ========== Training Utilities ==========
    EARLY_STOPPING = True
    PATIENCE = 10
    CHECKPOINT_DIR = 'checkpoints/'
    
    # ========== Logging ==========
    PRINT_EVERY = 1
    PLOT_TRAINING = True
    
    @classmethod
    def get_rng(cls):
        return np.random.default_rng(cls.RANDOM_SEED)
    
    @classmethod
    def set_seed(cls):
        np.random.seed(cls.RANDOM_SEED)
        print(f"🔒 Random seed set to {cls.RANDOM_SEED}")