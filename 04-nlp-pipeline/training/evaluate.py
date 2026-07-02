import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def evaluate_sentiment(model, test_loader, device):
    """Evaluate sentiment model and print metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            preds = (outputs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Negative', 'Positive']))
    
    return accuracy


def generate_text(model, vocab, seed_text, max_length=50, temperature=0.8):
    """Generate text using the model"""
    print(f"\n📝 Generating text from: '{seed_text}'")
    
    generated = model.generate(seed_text, vocab, max_length, temperature)
    
    print(f"✅ Generated: '{generated}'")
    
    return generated