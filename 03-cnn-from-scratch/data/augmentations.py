"""Simple, dependency-free data augmentation functions for image batches
shaped (N, C, H, W).
"""

import numpy as np


def random_horizontal_flip(x, p=0.5):
    out = x.copy()
    for i in range(x.shape[0]):
        if np.random.rand() < p:
            out[i] = out[i, :, :, ::-1]
    return out


def random_rotation_90(x, p=0.5):
    """Randomly rotates each sample by 90, 180, or 270 degrees."""
    out = x.copy()
    for i in range(x.shape[0]):
        if np.random.rand() < p:
            k = np.random.randint(1, 4)
            out[i] = np.rot90(out[i], k, axes=(1, 2))
    return out


def add_gaussian_noise(x, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, size=x.shape)
    return np.clip(x + noise, 0.0, 1.0)


def random_crop_pad(x, pad=2):
    """Pads the image then randomly crops back to the original size, which
    effectively performs a random translation.
    """
    N, C, H, W = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros_like(x)
    for i in range(N):
        top = np.random.randint(0, 2 * pad + 1)
        left = np.random.randint(0, 2 * pad + 1)
        out[i] = padded[i, :, top:top + H, left:left + W]
    return out


def augment_batch(x, flip=True, rotate=False, noise=True, crop=True):
    """Applies a standard augmentation pipeline. Rotation is off by default
    since it's a poor augmentation for digit datasets like MNIST (a rotated
    6 can look like a 9).
    """
    if flip:
        x = random_horizontal_flip(x)
    if rotate:
        x = random_rotation_90(x)
    if crop:
        x = random_crop_pad(x)
    if noise:
        x = add_gaussian_noise(x)
    return x