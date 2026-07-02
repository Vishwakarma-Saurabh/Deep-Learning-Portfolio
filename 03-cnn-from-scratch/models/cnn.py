"""Sequential CNN model built from the scratch layers."""

import pickle

from layers.conv2d import Conv2D
from layers.pooling import MaxPool2D
from layers.flatten import Flatten
from layers.dense import Dense
from layers.activation import ReLU, Softmax
from layers.batchnorm import BatchNorm2D
from layers.dropout import Dropout


class CNN:
    """A simple sequential container that chains layers together."""

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def set_training(self, mode=True):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode

    def get_params_and_grads(self):
        """Returns a flat list of (param_array, grad_array) tuples for the optimizer."""
        all_params = []
        for layer in self.layers:
            params, grads = layer.params_and_grads()
            for key in params:
                all_params.append((params[key], grads[key]))
        return all_params

    def predict(self, x):
        was_training = getattr(self.layers[0], 'training', None)
        self.set_training(False)
        out = self.forward(x)
        if was_training is not None:
            self.set_training(True)
        return out

    def save(self, path):
        state = []
        for layer in self.layers:
            params, _ = layer.params_and_grads()
            state.append({k: v.copy() for k, v in params.items()})
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        for layer, saved_params in zip(self.layers, state):
            params, _ = layer.params_and_grads()
            for key in params:
                params[key][...] = saved_params[key]


def build_simple_cnn(input_channels=1, num_classes=10):
    """Builds a small CNN for 28x28 images (e.g. MNIST):

    Conv-BN-ReLU-Pool -> Conv-BN-ReLU-Pool -> Flatten -> Dense-ReLU-Dropout -> Dense-Softmax
    """
    model = CNN()
    model.add(Conv2D(input_channels, 8, kernel_size=3, stride=1, padding=1))
    model.add(BatchNorm2D(8))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))  # 28x28 -> 14x14

    model.add(Conv2D(8, 16, kernel_size=3, stride=1, padding=1))
    model.add(BatchNorm2D(16))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))  # 14x14 -> 7x7

    model.add(Flatten())
    model.add(Dense(16 * 7 * 7, 64))
    model.add(ReLU())
    model.add(Dropout(p=0.5))
    model.add(Dense(64, num_classes))
    model.add(Softmax())
    return model