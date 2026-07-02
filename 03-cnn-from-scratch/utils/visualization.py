import os

import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    OUTPUT_DIR = 'outputs'

    @classmethod
    def _save_or_show(cls, fig, filename):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        path = os.path.join(cls.OUTPUT_DIR, filename)
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {path}")

    @staticmethod
    def plot_training_history(history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(history['lr'], color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time
        axes[1, 1].plot(np.cumsum(history['time']), linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Cumulative Time (s)')
        axes[1, 1].set_title('Training Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        Visualizer._save_or_show(fig, 'training_history.png')
    
    @staticmethod
    def plot_conv_weights(model, num_filters=16):
        """Visualize convolutional kernels"""
        kernels = None
        for layer in model.layers:
            if hasattr(layer, 'kernels'):
                kernels = layer.kernels
                break
        
        if kernels is None:
            print("No convolutional layers found")
            return
        
        # Plot first 16 kernels
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        
        for i in range(min(num_filters, kernels.shape[0])):
            # Average across input channels
            kernel = kernels[i].mean(axis=0)
            axes[i].imshow(kernel, cmap='RdBu_r')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i+1}')
        
        plt.suptitle('Convolutional Filters')
        plt.tight_layout()
        Visualizer._save_or_show(fig, 'conv_weights.png')
    
    @staticmethod
    def show_predictions(model, X, y, num_samples=12):
        """Show sample predictions"""
        predictions = model.predict(X)
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img = X[idx, 0]  # Remove channel dimension
            true_label = y[idx]
            pred_label = predictions[idx]
            confidence = np.max(model.forward(X[idx:idx+1], training=False))
            
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}',
                color=color
            )
        
        plt.tight_layout()
        plt.show()