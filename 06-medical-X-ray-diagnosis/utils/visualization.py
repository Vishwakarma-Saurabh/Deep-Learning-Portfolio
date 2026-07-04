import matplotlib.pyplot as plt
import torch
import numpy as np
from config import Config

def plot_training_history(history, title, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy (if available)
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{title} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
    
    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            probs, _ = model(images)
            preds = (probs > 0.5).float()
            
            # Show image
            img = images[0].cpu().permute(1, 2, 0).numpy()
            axes[idx, 0].imshow(img, cmap='gray')
            axes[idx, 0].set_title('X-ray Image')
            axes[idx, 0].axis('off')
            
            # Show predictions
            diseases = []
            for i, (pred, prob) in enumerate(zip(preds[0], probs[0])):
                if pred == 1:
                    diseases.append(f"{Config.DISEASES[i]}: {prob.item():.2%}")
            
            text = "\n".join(diseases) if diseases else "No abnormalities detected"
            axes[idx, 1].text(0.1, 0.5, text, fontsize=10, verticalalignment='center')
            axes[idx, 1].set_title('Predictions')
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()