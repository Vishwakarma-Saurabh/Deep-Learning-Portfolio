import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from models.gan import GANLoss

def train_gan(generator, discriminator, train_loader, config):
    """Train GAN model"""
    
    device = config.DEVICE
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config.GAN_LEARNING_RATE,
        betas=(config.GAN_BETA1, 0.999)
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config.GAN_LEARNING_RATE,
        betas=(config.GAN_BETA1, 0.999)
    )
    
    gan_loss = GANLoss()
    history = {'g_loss': [], 'd_loss': []}
    
    print("=" * 60)
    print("🔄 TRAINING GAN")
    print("=" * 60)
    
    for epoch in range(config.GAN_EPOCHS):
        g_loss_total = 0
        d_loss_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.GAN_EPOCHS}")
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # ========== Train Discriminator ==========
            d_optimizer.zero_grad()
            
            # Real images
            real_pred = discriminator(real_images)
            d_loss_real = gan_loss.discriminator_loss(real_pred, torch.ones_like(real_pred))
            
            # Fake images
            z = torch.randn(batch_size, config.GAN_LATENT_DIM).to(device)
            fake_images = generator(z)
            fake_pred = discriminator(fake_images.detach())
            d_loss_fake = gan_loss.discriminator_loss(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, config.GAN_LATENT_DIM).to(device)
            fake_images = generator(z)
            fake_pred = discriminator(fake_images)
            g_loss = gan_loss.generator_loss(fake_pred)
            
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            })
        
        avg_g_loss = g_loss_total / len(train_loader)
        avg_d_loss = d_loss_total / len(train_loader)
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        
        print(f"Epoch {epoch+1}/{config.GAN_EPOCHS} | "
              f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 5 == 0:
            samples = generator(torch.randn(16, config.GAN_LATENT_DIM).to(device))
            save_samples(samples, epoch, config)
    
    return history


def save_samples(samples, epoch, config):
    """Save generated samples"""
    os.makedirs(f"{config.OUTPUT_DIR}/gan", exist_ok=True)
    
    samples = (samples + 1) / 2  # Denormalize
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = samples[i].cpu().detach().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/gan/epoch_{epoch+1}.png")
    plt.close()