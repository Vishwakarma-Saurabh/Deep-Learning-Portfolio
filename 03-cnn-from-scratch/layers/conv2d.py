"""2D convolution layer implemented from scratch with an im2col/col2im trick
so the forward and backward passes reduce to matrix multiplications instead
of slow Python loops over every output pixel.
"""

import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im(cols, x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    """Standard 2D convolution layer.

    Weights shape: (out_channels, in_channels, kernel_size, kernel_size)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        # He initialization, good default for ReLU networks
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x_cols = None
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        F = self.kernel_size
        out_h = (H + 2 * self.padding - F) // self.stride + 1
        out_w = (W + 2 * self.padding - F) // self.stride + 1

        self.x_cols = im2col(x, F, F, self.padding, self.stride)
        W_col = self.W.reshape(self.out_channels, -1)

        out = W_col @ self.x_cols
        if self.use_bias:
            out = out + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        N = dout.shape[0]
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.dW = (dout_reshaped @ self.x_cols.T).reshape(self.W.shape)
        if self.use_bias:
            self.db = np.sum(dout_reshaped, axis=1, keepdims=True)

        W_col = self.W.reshape(self.out_channels, -1)
        dx_cols = W_col.T @ dout_reshaped
        dx = col2im(dx_cols, self.x_shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        return dx

    def params_and_grads(self):
        params = {'W': self.W, 'b': self.b}
        grads = {'W': self.dW, 'b': self.db}
        return params, grads