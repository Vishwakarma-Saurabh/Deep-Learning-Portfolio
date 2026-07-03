import os
import torch
import matplotlib.pyplot as plt
from config import Config
from data import get_mnist_loaders
from models import VAE, Generator, Discriminator, DiffusionModel
from training import train_vae, train_gan, train_diffusion
from utils import plot_training_history, compare_models

def main():
    """Main entry point"""
    Config.set_seed()
    
    # Create directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    train_loader, test_loader = get_mnist_loaders(Config.VAE_BATCH_SIZE)
    
    print("=" * 60)
    print("🎨 GENERATIVE MODELS SHOWDOWN")
    print("=" * 60)
    
    # ============ VAE ============
    print("\n📦 Training VAE...")
    vae = VAE(latent_dim=Config.LATENT_DIM)
    vae_history = train_vae(vae, train_loader, Config)
    
    # Generate samples
    vae_samples = vae.generate(16)
    
    # ============ GAN ============
    print("\n🎭 Training GAN...")
    generator = Generator(latent_dim=Config.GAN_LATENT_DIM)
    discriminator = Discriminator()
    gan_history = train_gan(generator, discriminator, train_loader, Config)
    
    # Generate samples
    gan_samples = generator(torch.randn(16, Config.GAN_LATENT_DIM).to(Config.DEVICE))
    
    # ============ Diffusion ============
    print("\n🌀 Training Diffusion...")
    diffusion = DiffusionModel(
        timesteps=Config.DIFFUSION_TIMESTEPS,
        beta_start=Config.DIFFUSION_BETA_START,
        beta_end=Config.DIFFUSION_BETA_END
    )
    diffusion_history = train_diffusion(diffusion, train_loader, Config)
    
    # Generate samples
    diffusion_samples = diffusion.generate(16)
    
    # ============ Comparison ============
    print("\n" + "=" * 60)
    print("📊 COMPARING RESULTS")
    print("=" * 60)
    
    compare_models(vae_samples, gan_samples, diffusion_samples)
    
    # Plot histories
    plot_training_history(
        vae_history['loss'],
        'VAE Training Loss',
        f'{Config.OUTPUT_DIR}/vae_loss.png'
    )
    
    plot_training_history(
        gan_history,
        'GAN Training Loss',
        f'{Config.OUTPUT_DIR}/gan_loss.png'
    )
    
    plot_training_history(
        diffusion_history['loss'],
        'Diffusion Training Loss',
        f'{Config.OUTPUT_DIR}/diffusion_loss.png'
    )
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT 5 COMPLETED!")
    print("=" * 60)
    print("VAE: Lower quality, smooth latent space")
    print("GAN: Sharp images, unstable training")
    print("Diffusion: High quality, stable training")
    print("=" * 60)


if __name__ == "__main__":
    main()