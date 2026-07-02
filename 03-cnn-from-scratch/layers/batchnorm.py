import numpy as np


class BatchNorm2D:
    """Batch Normalization for 2D inputs (convolutional layers)."""

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((1, num_features, 1, 1), dtype=np.float32)
        self.beta = np.zeros((1, num_features, 1, 1), dtype=np.float32)

        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)

        self.cache = {}

    def forward(self, X, training=True):
        if training:
            mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            var = np.var(X, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.astype(np.float32)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.astype(np.float32)

            inv_std = 1.0 / np.sqrt(var + self.eps)
            X_norm = (X - mean) * inv_std
            self.cache = {'X': X, 'mean': mean, 'var': var, 'X_norm': X_norm, 'training': True}
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            X_norm = (X - self.running_mean) * inv_std
            self.cache = {'X': X, 'training': False}

        out = (self.gamma * X_norm + self.beta).astype(np.float32)
        self.cache['output'] = out
        return out

    def backward(self, grad):
        if not self.cache.get('training', False):
            return grad, np.zeros_like(self.gamma), np.zeros_like(self.beta)

        X = self.cache['X']
        X_norm = self.cache['X_norm']
        var = self.cache['var']
        m = grad.shape[0] * grad.shape[2] * grad.shape[3]

        dgamma = np.sum(grad * X_norm, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True).astype(np.float32)

        inv_std = 1.0 / np.sqrt(var + self.eps)
        dxhat = grad * self.gamma
        dX = (inv_std / m) * (
            m * dxhat
            - np.sum(dxhat, axis=(0, 2, 3), keepdims=True)
            - X_norm * np.sum(dxhat * X_norm, axis=(0, 2, 3), keepdims=True)
        )

        return dX.astype(np.float32), dgamma, dbeta
