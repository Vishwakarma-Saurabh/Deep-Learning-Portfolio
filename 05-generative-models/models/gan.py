import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """GAN Generator: Noise → Image"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 14x14
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 28x28
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)    # 28x28
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        return x


class Discriminator(nn.Module):
    """GAN Discriminator: Image → Real/Fake"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7x7
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 4x4
        
        self.fc = nn.Linear(128 * 4 * 4, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return torch.sigmoid(x)


class GAN(nn.Module):
    """Generative Adversarial Network"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim
    
    def generate(self, num_samples=16):
        """Generate new images from random noise"""
        z = torch.randn(num_samples, self.latent_dim).to(self.generator.fc.weight.device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples


class GANLoss:
    """GAN Loss functions"""
    @staticmethod
    def generator_loss(fake_pred):
        """Generator wants to fool discriminator"""
        return -torch.mean(torch.log(fake_pred + 1e-8))
    
    @staticmethod
    def discriminator_loss(real_pred, fake_pred):
        """Discriminator wants to correctly classify"""
        real_loss = -torch.mean(torch.log(real_pred + 1e-8))
        fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
        return real_loss + fake_loss