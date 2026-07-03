import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_vae(model, train_loader, config):
    """Train VAE model"""
    
    device = config.DEVICE
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.VAE_LEARNING_RATE)
    
    history = {'loss': []}
    
    print("=" * 60)
    print("🔄 TRAINING VAE")
    print("=" * 60)
    
    for epoch in range(config.VAE_EPOCHS):
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.VAE_EPOCHS}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Forward pass
            recon, mean, logvar = model(data)
            
            # ✅ FIXED: Use compute_loss method
            loss = model.compute_loss(recon, data, mean, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{config.VAE_EPOCHS} | Loss: {avg_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 5 == 0:
            samples = model.generate(16)
            save_samples(samples, epoch, config)
    
    return history


def save_samples(samples, epoch, config):
    """Save generated samples"""
    os.makedirs(f"{config.OUTPUT_DIR}/vae", exist_ok=True)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = samples[i].cpu().detach().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/vae/epoch_{epoch+1}.png")
    plt.close()