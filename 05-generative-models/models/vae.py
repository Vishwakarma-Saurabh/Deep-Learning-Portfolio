import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """VAE Encoder: Compresses image to latent distribution"""
    def __init__(self, latent_dim=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7x7
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 4x4
        
        self.fc_mean = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    """VAE Decoder: Reconstructs image from latent code"""
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 16x16
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)    # 32x32
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        
        # Crop to 28x28
        x = x[:, :, 2:30, 2:30]
        return x


class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick: z = mean + std * noise"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar
    
    def compute_loss(self, recon, x, mean, logvar):
        """
        Compute VAE loss with robust handling
        Handles both [0,1] and [-1,1] normalized data
        """
        # ✅ FIX: Clamp recon to [0,1] for BCE
        recon_clamped = torch.clamp(recon, 1e-7, 1 - 1e-7)
        
        # ✅ FIX: Scale x to [0,1] if it's in [-1,1]
        if x.min() < 0:  # Data is normalized to [-1, 1]
            x_scaled = (x + 1) / 2  # Scale to [0, 1]
        else:
            x_scaled = x
        
        # Reconstruction Loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_clamped, x_scaled, reduction='sum')
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        return (recon_loss + kl_loss) / x.size(0)
    
    def generate(self, num_samples=16):
        """Generate new images from random latent codes"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples


class VAELoss(nn.Module):
    """Alternative: VAE Loss as nn.Module"""
    def __init__(self):
        super().__init__()
    
    def forward(self, recon, x, mean, logvar):
        # Clamp for numerical stability
        recon = torch.clamp(recon, 1e-7, 1 - 1e-7)
        
        # Handle [-1, 1] vs [0, 1] normalization
        if x.min() < 0:
            x = (x + 1) / 2
        
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        return (recon_loss + kl_loss) / x.size(0)