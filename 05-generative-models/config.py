import torch

class Config:
    """Configuration for generative models"""
    
    # ========== Device ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== Data ==========
    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    LATENT_DIM = 20  # For VAE
    
    # ========== VAE ==========
    VAE_EPOCHS = 20
    VAE_BATCH_SIZE = 128
    VAE_LEARNING_RATE = 0.001
    
    # ========== GAN ==========
    GAN_EPOCHS = 50
    GAN_BATCH_SIZE = 128
    GAN_LEARNING_RATE = 0.0002
    GAN_LATENT_DIM = 100
    GAN_BETA1 = 0.5
    
    # ========== Diffusion ==========
    DIFFUSION_EPOCHS = 20
    DIFFUSION_BATCH_SIZE = 128
    DIFFUSION_LEARNING_RATE = 0.001
    DIFFUSION_TIMESTEPS = 1000
    DIFFUSION_BETA_START = 0.0001
    DIFFUSION_BETA_END = 0.02
    
    # ========== Paths ==========
    OUTPUT_DIR = 'outputs/'
    CHECKPOINT_DIR = 'checkpoints/'
    
    @classmethod
    def set_seed(cls, seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"🔒 Random seed set to {seed}")