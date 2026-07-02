"""Dataset loading utilities: MNIST downloader/parser, a synthetic-data
fallback (useful when there's no internet access), and a simple mini-batch
DataLoader.
"""

import gzip
import os
import urllib.request

import numpy as np

MNIST_URLS = {
    'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def _load_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)


def load_mnist(data_dir='./mnist_data'):
    """Downloads (if needed) and loads the MNIST dataset.

    Returns (x_train, y_train), (x_test, y_test) with images shaped
    (N, 1, 28, 28) scaled to [0, 1], and integer labels shaped (N,).
    """
    paths = {}
    for key, url in MNIST_URLS.items():
        path = os.path.join(data_dir, os.path.basename(url))
        _download(url, path)
        paths[key] = path

    x_train = _load_images(paths['train_images'])
    y_train = _load_labels(paths['train_labels'])
    x_test = _load_images(paths['test_images'])
    y_test = _load_labels(paths['test_labels'])

    return (x_train, y_train), (x_test, y_test)


def one_hot(labels, num_classes=10):
    out = np.zeros((labels.shape[0], num_classes))
    out[np.arange(labels.shape[0]), labels] = 1
    return out


def make_synthetic_dataset(num_samples=200, num_classes=10, channels=1, height=28, width=28, seed=0):
    """Generates random synthetic image data. Handy for quickly smoke-testing
    the pipeline without needing to download a real dataset.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, channels, height, width)).astype(np.float32)
    y = rng.integers(0, num_classes, size=num_samples)
    return x, y


class DataLoader:
    """Simple mini-batch iterator with optional shuffling."""

    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))