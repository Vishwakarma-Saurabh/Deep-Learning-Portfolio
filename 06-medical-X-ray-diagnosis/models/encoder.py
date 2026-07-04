import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """CNN Encoder for extracting image features"""
    
    def __init__(self, pretrained=True, embedding_size=2048):
        super().__init__()
        
        # Use DenseNet121 as backbone
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Remove classification head
        self.features = self.backbone.features
        
        # Adaptive pooling to get fixed size features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear layer to get embedding
        self.fc = nn.Linear(1024, embedding_size)
        
        self.embedding_size = embedding_size
    
    def forward(self, x):
        # Extract features
        features = self.features(x)  # (batch, 1024, 7, 7)
        
        # Pool
        pooled = self.pool(features)  # (batch, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 1024)
        
        # Project to embedding
        embedding = self.fc(pooled)  # (batch, embedding_size)
        
        return embedding, features