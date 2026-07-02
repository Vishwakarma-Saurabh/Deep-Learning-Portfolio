"""Activation function layers."""

import numpy as np


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params_and_grads(self):
        return {}, {}


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

    def params_and_grads(self):
        return {}, {}


class Softmax:
    """Numerically stable softmax.

    Note: when paired with cross-entropy loss, the combined gradient is
    typically computed directly as (probs - one_hot_labels) for simplicity
    and numerical stability (see training/trainer.py). backward() here is a
    passthrough so the layer still behaves correctly if used standalone.
    """

    def __init__(self):
        self.out = None

    def forward(self, x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        return dout

    def params_and_grads(self):
        return {}, {}