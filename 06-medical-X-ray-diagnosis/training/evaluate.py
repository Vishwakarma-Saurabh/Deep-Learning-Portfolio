import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def evaluate_classifier(model, test_loader, device):
    """Evaluate multi-label classifier"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            probs, _ = model(images)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten()) * 100
    
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
    except:
        auc = 0
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n📊 Classifier Evaluation:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return accuracy, auc, f1


def evaluate_report_generator(model, test_loader, vocab, device, max_length=100):
    """Evaluate report generator"""
    model.eval()
    
    generated_reports = []
    actual_reports = []
    
    idx_to_word = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        for images, _, captions in test_loader[:10]:  # Only 10 samples
            images = images.to(device)
            
            reports = model.generate_report(
                images,
                vocab,
                idx_to_word,
                max_length
            )
            generated_reports.extend(reports)
    
    print("\n📝 Sample Generated Reports:")
    for i, report in enumerate(generated_reports[:5]):
        print(f"  Report {i+1}: {report}")
    
    return generated_reports