import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_history(history, save_path=None):
    """Plot training loss and accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def predict_sentiment(text, model, vocab, device, max_length=200):
    """Predict sentiment of a single text"""
    model.eval()
    
    # Preprocess
    tokens = vocab.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        sentiment = "Positive" if probability > 0.5 else "Negative"
    
    print(f"\n📝 Text: {text}")
    print(f"   Sentiment: {sentiment} ({probability:.2%} confidence)")
    
    return sentiment, probability