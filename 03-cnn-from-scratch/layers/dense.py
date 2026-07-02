"""Fully connected (dense) layer."""

import numpy as np


class Dense:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx

    def params_and_grads(self):
        params = {'W': self.W, 'b': self.b}
        grads = {'W': self.dW, 'b': self.db}
        return params, grads