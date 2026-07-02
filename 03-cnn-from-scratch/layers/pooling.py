"""Max pooling layer implemented from scratch."""

import numpy as np


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.max_indices = None  # (N, C, out_h, out_w, 2) -> (row, col) within each window

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        p = self.pool_size
        s = self.stride
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1

        out = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i * s, i * s + p
                w_start, w_end = j * s, j * s + p
                window = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

                window_reshaped = window.reshape(N, C, -1)
                idx = np.argmax(window_reshaped, axis=2)
                self.max_indices[:, :, i, j, 0] = idx // p
                self.max_indices[:, :, i, j, 1] = idx % p
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        p = self.pool_size
        s = self.stride
        out_h, out_w = dout.shape[2], dout.shape[3]
        dx = np.zeros_like(self.x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                hi = self.max_indices[:, :, i, j, 0]
                wi = self.max_indices[:, :, i, j, 1]
                for n in range(N):
                    for c in range(C):
                        dx[n, c, h_start + hi[n, c], w_start + wi[n, c]] += dout[n, c, i, j]
        return dx

    def params_and_grads(self):
        return {}, {}