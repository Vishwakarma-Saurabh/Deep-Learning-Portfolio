"""Plotting helpers for training curves, learned filters, sample
predictions, and a confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='train_loss')
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='train_acc')
    if history.get('val_acc'):
        axes[1].plot(history['val_acc'], label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_filters(conv_layer, save_path=None):
    """Visualizes the learned filters of a Conv2D layer (first input channel only)."""
    weights = conv_layer.W  # (out_channels, in_channels, kh, kw)
    num_filters = weights.shape[0]
    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis('off')
        if i < num_filters:
            ax.imshow(weights[i, 0], cmap='gray')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, num_classes=10, save_path=None):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, matrix[i, j], ha='center', va='center',
                     color='white' if matrix[i, j] > matrix.max() / 2 else 'black')

    fig.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def show_sample_predictions(x, y_true, y_pred, num_samples=8, save_path=None):
    num_samples = min(num_samples, x.shape[0])
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 2))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        ax.axis('off')
        img = x[i, 0] if x.shape[1] == 1 else np.transpose(x[i], (1, 2, 0))
        ax.imshow(img, cmap='gray' if x.shape[1] == 1 else None)
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}", color=color, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)