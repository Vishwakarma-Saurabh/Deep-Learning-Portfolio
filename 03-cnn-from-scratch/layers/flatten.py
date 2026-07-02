"""Flatten layer: reshapes (N, C, H, W) feature maps into (N, C*H*W) vectors."""


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

    def params_and_grads(self):
        return {}, {}