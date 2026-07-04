import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import ReportDecoder

class ReportGenerator(nn.Module):
    """Complete Report Generation System (Encoder + Decoder)"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, feature_dim=1024):
        super().__init__()
        
        self.encoder = ImageEncoder(embedding_size=feature_dim)
        self.decoder = ReportDecoder(vocab_size, embed_dim, hidden_dim, feature_dim)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
    
    def forward(self, images, captions):
        # Encode images
        features, raw_features = self.encoder(images)
        
        # Get spatial features
        batch_size = images.size(0)
        spatial_features = raw_features.permute(0, 2, 3, 1)  # (batch, 7, 7, 1024)
        
        # Decode to report
        outputs = self.decoder(spatial_features, captions)
        
        return outputs
    
    def generate_report(self, images, vocab, idx_to_word, max_length=100):
        """Generate report from images"""
        features, raw_features = self.encoder(images)
        spatial_features = raw_features.permute(0, 2, 3, 1)
        reports = self.decoder.generate(
            spatial_features,
            vocab,
            idx_to_word,
            max_length
        )
        return reports