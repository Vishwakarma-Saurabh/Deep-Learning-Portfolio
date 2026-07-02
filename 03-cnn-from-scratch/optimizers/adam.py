"""Adam optimizer implemented from scratch."""

import numpy as np


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_and_grads):
        """params_and_grads: list of (param_array, grad_array) tuples.
        Updates each param_array in place.
        """
        self.t += 1
        for idx, (param, grad) in enumerate(params_and_grads):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(param)
                self.v[idx] = np.zeros_like(param)

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)