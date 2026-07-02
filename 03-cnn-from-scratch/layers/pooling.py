import numpy as np


class MaxPool2D:
    """Max Pooling layer from scratch."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def forward(self, X, training=True):
        batch_size, channels, in_h, in_w = X.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        if self.stride == self.pool_size:
            windows = X.reshape(
                batch_size,
                channels,
                out_h,
                self.pool_size,
                out_w,
                self.pool_size,
            )
            flat_windows = windows.reshape(batch_size, channels, out_h, out_w, -1)
            max_idx = np.argmax(flat_windows, axis=-1)
            output = np.max(flat_windows, axis=-1)
        else:
            output = np.zeros((batch_size, channels, out_h, out_w), dtype=X.dtype)
            max_idx = np.zeros((batch_size, channels, out_h, out_w), dtype=np.int64)
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    window = X[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    flat = window.reshape(batch_size, channels, -1)
                    max_idx[:, :, i, j] = np.argmax(flat, axis=-1)
                    output[:, :, i, j] = np.max(flat, axis=-1)

        self.cache = {
            'X_shape': X.shape,
            'max_idx': max_idx,
            'out_h': out_h,
            'out_w': out_w,
        }
        return output

    def backward(self, grad_output):
        X_shape = self.cache['X_shape']
        max_idx = self.cache['max_idx']
        out_h = self.cache['out_h']
        out_w = self.cache['out_w']

        batch_size, channels, in_h, in_w = X_shape
        dX = np.zeros(X_shape, dtype=grad_output.dtype)
        max_h, max_w = np.unravel_index(max_idx, (self.pool_size, self.pool_size))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                dX[:, :, h_start + max_h[:, :, i, j], w_start + max_w[:, :, i, j]] += grad_output[:, :, i, j]

        return dX


class AveragePool2D:
    """Average Pooling layer from scratch."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def forward(self, X, training=True):
        batch_size, channels, in_h, in_w = X.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        if self.stride == self.pool_size:
            windows = X.reshape(
                batch_size,
                channels,
                out_h,
                self.pool_size,
                out_w,
                self.pool_size,
            )
            output = windows.mean(axis=(3, 5))
        else:
            output = np.zeros((batch_size, channels, out_h, out_w), dtype=X.dtype)
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    window = X[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    output[:, :, i, j] = window.mean(axis=(2, 3))

        self.cache = {'X_shape': X.shape, 'out_h': out_h, 'out_w': out_w}
        return output

    def backward(self, grad_output):
        X_shape = self.cache['X_shape']
        out_h = self.cache['out_h']
        out_w = self.cache['out_w']

        dX = np.zeros(X_shape, dtype=grad_output.dtype)
        scale = grad_output / float(self.pool_size ** 2)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                dX[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size] += scale[:, :, i, j, None, None]

        return dX
