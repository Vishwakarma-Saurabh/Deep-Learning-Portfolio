import matplotlib.pyplot as plt
import torch
import os

def plot_training_history(history, title, save_path=None):
    """Plot training loss"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(history, dict):
        for key, values in history.items():
            ax.plot(values, label=key, linewidth=2)
        ax.legend()
    else:
        ax.plot(history, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    plt.show()


def compare_models(vae_samples, gan_samples, diffusion_samples):
    """Compare samples from all three models"""
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    # VAE samples
    for i in range(8):
        axes[0, i].imshow(vae_samples[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('VAE', fontsize=12)
    
    # GAN samples
    for i in range(8):
        gan_img = (gan_samples[i] + 1) / 2
        axes[1, i].imshow(gan_img.squeeze().cpu(), cmap='gray')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('GAN', fontsize=12)
    
    # Diffusion samples
    for i in range(8):
        diff_img = (diffusion_samples[i] + 1) / 2
        axes[2, i].imshow(diff_img.squeeze().cpu(), cmap='gray')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Diffusion', fontsize=12)
    
    plt.tight_layout()
    plt.show()