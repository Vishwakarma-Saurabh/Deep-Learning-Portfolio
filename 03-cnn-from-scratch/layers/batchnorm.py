"""Batch normalization for 2D feature maps (per-channel statistics)."""

import numpy as np


class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.9):
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))

        self.x_norm = None
        self.x_centered = None
        self.std_inv = None
        self.training = True

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.x_centered = x - mean
            self.std_inv = 1.0 / np.sqrt(var + self.eps)
            self.x_norm = self.x_centered * self.std_inv

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        N, C, H, W = dout.shape
        m = N * H * W

        self.dgamma = np.sum(dout * self.x_norm, axis=(0, 2, 3), keepdims=True)
        self.dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        dx_norm = dout * self.gamma
        dvar_term = np.sum(dx_norm * self.x_centered, axis=(0, 2, 3), keepdims=True)
        dx = (1.0 / m) * self.std_inv * (
            m * dx_norm
            - np.sum(dx_norm, axis=(0, 2, 3), keepdims=True)
            - self.x_centered * (self.std_inv ** 2) * dvar_term
        )
        return dx

    def params_and_grads(self):
        params = {'gamma': self.gamma, 'beta': self.beta}
        grads = {'gamma': self.dgamma, 'beta': self.dbeta}
        return params, grads