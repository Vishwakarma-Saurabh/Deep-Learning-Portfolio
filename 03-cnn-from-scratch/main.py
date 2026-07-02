"""Entry point: trains a from-scratch CNN on MNIST (falls back to synthetic
data if the dataset can't be downloaded, e.g. no internet access), then
saves the model and generates plots.

Usage:
    python main.py
"""

import os

import numpy as np

import config
from models.cnn import build_simple_cnn
from optimizers.adam import Adam
from optimizers.sgd import SGD
from data.dataset import load_mnist, make_synthetic_dataset, DataLoader
from training.trainer import Trainer
from utils.visualization import (
    plot_history,
    plot_filters,
    plot_confusion_matrix,
    show_sample_predictions,
)


def get_optimizer():
    if config.OPTIMIZER == 'adam':
        return Adam(learning_rate=config.LEARNING_RATE)
    return SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM)


def load_data():
    try:
        (x_train, y_train), (x_test, y_test) = load_mnist(config.DATA_DIR)
        print(f"Loaded MNIST: train={x_train.shape}, test={x_test.shape}")
    except Exception as e:
        print(f"Could not download MNIST ({e}); using synthetic data instead.")
        x_train, y_train = make_synthetic_dataset(
            num_samples=512, num_classes=config.NUM_CLASSES,
            channels=config.INPUT_CHANNELS, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE,
        )
        x_test, y_test = make_synthetic_dataset(
            num_samples=128, num_classes=config.NUM_CLASSES,
            channels=config.INPUT_CHANNELS, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, seed=1,
        )
    return (x_train, y_train), (x_test, y_test)


def main():
    np.random.seed(config.RANDOM_SEED)
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_data()

    train_loader = DataLoader(x_train, y_train, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_simple_cnn(input_channels=config.INPUT_CHANNELS, num_classes=config.NUM_CLASSES)
    optimizer = get_optimizer()
    trainer = Trainer(model, optimizer)

    print("Starting training...")
    history = trainer.fit(train_loader, val_loader=test_loader, epochs=config.EPOCHS)

    model.save(config.CHECKPOINT_PATH)
    print(f"Model saved to {config.CHECKPOINT_PATH}")

    plot_history(history, save_path=os.path.join(config.PLOTS_DIR, 'history.png'))
    plot_filters(model.layers[0], save_path=os.path.join(config.PLOTS_DIR, 'filters.png'))

    # Sample predictions on one test batch
    x_batch, y_batch = next(iter(test_loader))
    probs = model.predict(x_batch)
    preds = np.argmax(probs, axis=1)
    show_sample_predictions(x_batch, y_batch, preds, save_path=os.path.join(config.PLOTS_DIR, 'samples.png'))

    # Full confusion matrix over the test set
    all_preds, all_labels = [], []
    for x_b, y_b in test_loader:
        p = model.predict(x_b)
        all_preds.extend(np.argmax(p, axis=1))
        all_labels.extend(y_b)
    plot_confusion_matrix(
        all_labels, all_preds, num_classes=config.NUM_CLASSES,
        save_path=os.path.join(config.PLOTS_DIR, 'confusion_matrix.png'),
    )
    print(f"Plots saved to {config.PLOTS_DIR}/")


if __name__ == '__main__':
    main()