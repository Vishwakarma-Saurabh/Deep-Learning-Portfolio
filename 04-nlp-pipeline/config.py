import torch

class Config:
    """Configuration for NLP pipeline"""
    
    # ========== Device ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== Data ==========
    MAX_VOCAB_SIZE = 25000
    MAX_SEQUENCE_LENGTH = 200
    BATCH_SIZE = 64
    
    # ========== Embedding ==========
    EMBEDDING_DIM = 100
    
    # ========== LSTM/GRU ==========
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    
    # ========== Training ==========
    EPOCHS = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # ========== Text Generation ==========
    GENERATION_SEQUENCE_LENGTH = 50
    TEMPERATURE = 0.8  # Controls randomness (0.5-1.0)
    
    # ========== Paths ==========
    DATA_DIR = 'data/'
    CHECKPOINT_DIR = 'checkpoints/'
    OUTPUT_DIR = 'outputs/'
    
    @classmethod
    def set_seed(cls, seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"🔒 Random seed set to {seed}")