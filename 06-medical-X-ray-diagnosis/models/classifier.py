import torch
import torch.nn as nn
import torchvision.models as models

class ChestXRayClassifier(nn.Module):
    """Multi-label disease classifier for chest X-rays"""
    
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Use DenseNet121 backbone
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier.in_features
        
        # Replace classifier with multi-label head
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (batch, feature_dim)
        
        # Multi-label classification
        logits = self.classifier(features)  # (batch, num_classes)
        probs = self.sigmoid(logits)  # (batch, num_classes)
        
        return probs, features
    
    def predict(self, x, threshold=0.5):
        """Predict diseases with threshold"""
        probs, features = self.forward(x)
        predictions = (probs > threshold).float()
        return predictions, probs, features