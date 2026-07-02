import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, rng=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        self.rng = rng if rng is not None else np.random.default_rng(42)

        self.kernels = self.rng.normal(
            0,
            np.sqrt(2.0 / (in_channels * self.kernel_h * self.kernel_w)),
            size=(out_channels, in_channels, self.kernel_h, self.kernel_w),
        ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.cache = {}

    def _output_shape(self, height, width):
        out_h = (height + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_w) // self.stride + 1
        return out_h, out_w

    def forward(self, X, training=True):
        batch_size, in_channels, in_h, in_w = X.shape
        self.cache['X'] = X

        if self.padding > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            X_padded = X

        out_h, out_w = self._output_shape(in_h, in_w)
        windows = sliding_window_view(X_padded, (self.kernel_h, self.kernel_w), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]

        output = np.einsum('biyxkl,oikl->boyx', windows, self.kernels, optimize=True)
        output = output.astype(np.float32, copy=False)
        output += self.bias[None, :, None, None]

        self.cache['output'] = output
        self.cache['X_padded'] = X_padded
        self.cache['windows'] = windows
        return output

    def backward(self, grad_output):
        X = self.cache['X']
        X_padded = self.cache['X_padded']
        windows = self.cache['windows']
        batch_size, in_channels, in_h, in_w = X.shape
        _, _, out_h, out_w = grad_output.shape

        dK = np.einsum('boyx,biyxkl->oikl', grad_output, windows, optimize=True) / batch_size
        db = np.sum(grad_output, axis=(0, 2, 3)) / batch_size

        dX_padded = np.zeros_like(X_padded)
        for oc in range(self.out_channels):
            for ic in range(in_channels):
                kernel = self.kernels[oc, ic]
                for kh in range(self.kernel_h):
                    for kw in range(self.kernel_w):
                        h_slice = slice(kh, kh + out_h * self.stride, self.stride)
                        w_slice = slice(kw, kw + out_w * self.stride, self.stride)
                        dX_padded[:, ic, h_slice, w_slice] += grad_output[:, oc] * kernel[kh, kw]

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        return dX, dK, db
