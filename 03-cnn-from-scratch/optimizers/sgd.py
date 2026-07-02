"""Stochastic Gradient Descent optimizer, with optional momentum."""

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, params_and_grads):
        """params_and_grads: list of (param_array, grad_array) tuples.
        Updates each param_array in place.
        """
        for idx, (param, grad) in enumerate(params_and_grads):
            if self.momentum > 0:
                if idx not in self.velocities:
                    self.velocities[idx] = np.zeros_like(param)
                self.velocities[idx] = self.momentum * self.velocities[idx] - self.lr * grad
                param += self.velocities[idx]
            else:
                param -= self.lr * grad