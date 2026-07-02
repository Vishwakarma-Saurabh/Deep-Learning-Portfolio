"""Inverted dropout layer."""

import numpy as np


class Dropout:
    def __init__(self, p=0.5):
        """p: probability of dropping a unit (setting it to zero)."""
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        if self.training:
            return dout * self.mask
        return dout

    def params_and_grads(self):
        return {}, {}