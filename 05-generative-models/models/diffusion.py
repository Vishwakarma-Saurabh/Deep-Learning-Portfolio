import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """Embed time step for diffusion model"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class UNetBlock(nn.Module):
    """Basic UNet block with time embedding"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x, t_emb):
        x = F.relu(self.bn1(self.conv1(x)))
        time_effect = self.time_mlp(t_emb)[:, :, None, None]
        x = x + time_effect
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """U-Net architecture for diffusion model"""
    def __init__(self, in_channels=1, base_dim=64, time_dim=256):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, base_dim, time_dim)
        self.enc2 = UNetBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = UNetBlock(base_dim * 2, base_dim * 4, time_dim)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_dim * 4, base_dim * 8, time_dim)
        
        # Decoder
        self.dec3 = UNetBlock(base_dim * 8 + base_dim * 4, base_dim * 4, time_dim)
        self.dec2 = UNetBlock(base_dim * 4 + base_dim * 2, base_dim * 2, time_dim)
        self.dec1 = UNetBlock(base_dim * 2 + base_dim, base_dim, time_dim)
        
        # Output
        self.out_conv = nn.Conv2d(base_dim, in_channels, 1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Down/Up sample
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(self.down(e3), t_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        
        return self.out_conv(d1)


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Noise schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # UNet
        self.unet = UNet()
    
    def forward(self, x_0):
        """Training: Add noise and predict it"""
        t = torch.randint(0, self.timesteps, (x_0.size(0),), device=x_0.device)
        
        # Add noise
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        predicted_noise = self.unet(x_t, t)
        
        return predicted_noise, noise
    
    @torch.no_grad()
    def sample(self, num_samples=16, image_size=28):
        """Generate images from noise"""
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(num_samples, 1, image_size, image_size, device=device)
        
        # Gradually denoise
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.unet(x, t_tensor)
            
            # Denoise step
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if t > 0:
                beta_t = self.beta[t]
                z = torch.randn_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                ) + torch.sqrt(beta_t) * z
            else:
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                )
        
        return x
    
    @torch.no_grad()
    def generate(self, num_samples=16):
        """Generate new images"""
        return self.sample(num_samples)