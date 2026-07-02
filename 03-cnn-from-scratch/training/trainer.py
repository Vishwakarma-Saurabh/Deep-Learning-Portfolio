"""Training loop, loss functions, and metrics."""

import time

import numpy as np


def cross_entropy_loss(probs, labels):
    """probs: (N, num_classes) softmax outputs. labels: (N,) integer class labels."""
    N = probs.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0)
    log_likelihood = -np.log(clipped[np.arange(N), labels])
    return float(np.sum(log_likelihood) / N)


def softmax_cross_entropy_grad(probs, labels):
    """Gradient of (softmax + cross-entropy) combined, w.r.t. the softmax
    layer's input logits. This is the standard `probs - one_hot` trick,
    which is simpler and more numerically stable than differentiating
    through softmax and cross-entropy separately.
    """
    N = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(N), labels] -= 1
    grad /= N
    return grad


def accuracy(probs, labels):
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == labels))


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader):
        self.model.set_training(True)
        losses, accs = [], []

        for x_batch, y_batch in train_loader:
            probs = self.model.forward(x_batch)
            loss = cross_entropy_loss(probs, y_batch)
            acc = accuracy(probs, y_batch)

            dout = softmax_cross_entropy_grad(probs, y_batch)
            self.model.backward(dout)

            self.optimizer.step(self.model.get_params_and_grads())

            losses.append(loss)
            accs.append(acc)

        return float(np.mean(losses)), float(np.mean(accs))

    def evaluate(self, val_loader):
        self.model.set_training(False)
        losses, accs = [], []

        for x_batch, y_batch in val_loader:
            probs = self.model.forward(x_batch)
            losses.append(cross_entropy_loss(probs, y_batch))
            accs.append(accuracy(probs, y_batch))

        self.model.set_training(True)
        return float(np.mean(losses)), float(np.mean(accs))

    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            log = f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                log += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"

            log += f" - time: {time.time() - start:.1f}s"

            if verbose:
                print(log)

        return self.history