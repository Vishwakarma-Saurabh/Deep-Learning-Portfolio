import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import pandas as pd
from data.preprocessing import preprocess_image, encode_report

class ChestXRayDataset(Dataset):
    """Synthetic Chest X-ray dataset for demonstration"""
    
    def __init__(self, num_samples=1000, is_train=True, vocab=None, max_length=100):
        self.num_samples = num_samples
        self.is_train = is_train
        self.vocab = vocab
        self.max_length = max_length
        
        # Generate synthetic data
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic X-ray images and reports"""
        self.images = []
        self.labels = []
        self.reports = []
        
        for i in range(self.num_samples):
            # Generate random image (synthetic X-ray)
            # In real project: load actual X-ray images
            img = np.random.rand(3, 224, 224).astype(np.float32)
            self.images.append(img)
            
            # Generate synthetic labels (multi-label)
            labels = np.random.randint(0, 2, 14).astype(np.float32)
            self.labels.append(labels)
            
            # Generate synthetic report based on labels
            report = self._generate_report(labels)
            self.reports.append(report)
    
    def _generate_report(self, labels):
        """Generate synthetic radiology report based on labels"""
        from config import Config
        
        findings = []
        
        # Find positive diseases
        positive_diseases = []
        for i, label in enumerate(labels):
            if label == 1:
                disease = Config.DISEASES[i]
                positive_diseases.append(disease)
        
        if not positive_diseases:
            return "Normal chest X-ray. No acute cardiopulmonary abnormalities identified."
        
        # Build report
        report = "Findings: "
        
        if 'Cardiomegaly' in positive_diseases:
            report += "Cardiomegaly is present with an enlarged cardiac silhouette. "
        
        if 'Pneumonia' in positive_diseases:
            report += "There is evidence of pneumonia with opacities in the lung fields. "
        
        if 'Effusion' in positive_diseases:
            report += "Pleural effusion is noted. "
        
        if 'Atelectasis' in positive_diseases:
            report += "Mild atelectasis is observed in the lung bases. "
        
        if 'Edema' in positive_diseases:
            report += "Pulmonary edema is present with interstitial markings. "
        
        if 'Emphysema' in positive_diseases:
            report += "Findings consistent with emphysema are noted. "
        
        if 'Mass' in positive_diseases:
            report += "A mass is visible in the lung parenchyma. "
        
        if 'Nodule' in positive_diseases:
            report += "A pulmonary nodule is identified. "
        
        # Add impression
        report += "Impression: "
        if len(positive_diseases) == 1:
            report += f"{positive_diseases[0]} is present."
        else:
            report += "Multiple findings are present: " + ", ".join(positive_diseases) + "."
        
        return report
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Image is already in (C, H, W) format
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Encode report
        if self.vocab:
            from data.preprocessing import encode_report
            report_tokens = encode_report(
                self.reports[idx],
                self.vocab,
                self.max_length
            )
            return image, labels, report_tokens
        
        return image, labels


def get_dataloaders(config, vocab=None):
    """Create train and test data loaders"""
    # Create datasets
    train_dataset = ChestXRayDataset(
        num_samples=1000,
        is_train=True,
        vocab=vocab,
        max_length=config.REPORT_MAX_LENGTH
    )
    
    test_dataset = ChestXRayDataset(
        num_samples=200,
        is_train=False,
        vocab=vocab,
        max_length=config.REPORT_MAX_LENGTH
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, test_loader