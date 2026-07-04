import torch

class Config:
    """Configuration for Medical Chest X-ray System"""
    
    # ========== Device ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== Data ==========
    IMAGE_SIZE = 224
    IMAGE_CHANNELS = 3
    NUM_CLASSES = 14  # NIH ChestX-ray14 diseases
    
    # Disease labels
    DISEASES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # ========== Model - Classifier ==========
    CLASSIFIER_EPOCHS = 10
    CLASSIFIER_BATCH_SIZE = 32
    CLASSIFIER_LEARNING_RATE = 0.0001
    CLASSIFIER_PRETRAINED = True
    
    # ========== Model - Report Generator ==========
    REPORT_EPOCHS = 10
    REPORT_BATCH_SIZE = 32
    REPORT_LEARNING_RATE = 0.001
    REPORT_HIDDEN_DIM = 512
    REPORT_EMBED_DIM = 256
    REPORT_MAX_LENGTH = 100
    
    # ========== Vocabulary ==========
    MAX_VOCAB_SIZE = 5000
    START_TOKEN = '<SOS>'
    END_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'
    
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