import numpy as np


class DataAugmentation:
    """Lightweight augmentation for images."""

    def __init__(self, rotation_range=10, shift_range=0.1, zoom_range=0.1, rng=None):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def __call__(self, X, y=None):
        if X.ndim == 4:
            batch = X.copy()
        else:
            batch = X[None, ...]

        if self.rotation_range <= 0 and self.shift_range <= 0 and self.zoom_range <= 0:
            return batch

        augmented = np.zeros_like(batch, dtype=np.float32)
        for i in range(batch.shape[0]):
            img = batch[i, 0]

            if self.shift_range > 0:
                shift_y = int(round(self.rng.uniform(-self.shift_range, self.shift_range) * 28))
                shift_x = int(round(self.rng.uniform(-self.shift_range, self.shift_range) * 28))
                img = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))

            if self.zoom_range > 0:
                zoom_factor = self.rng.uniform(1 - self.zoom_range, 1 + self.zoom_range)
                if zoom_factor != 1.0:
                    new_h = max(1, int(round(img.shape[0] * zoom_factor)))
                    new_w = max(1, int(round(img.shape[1] * zoom_factor)))
                    if new_h != img.shape[0] or new_w != img.shape[1]:
                        y_coords = np.linspace(0, img.shape[0] - 1, new_h)
                        x_coords = np.linspace(0, img.shape[1] - 1, new_w)
                        img = np.take(np.take(img, np.clip(y_coords.astype(int), 0, img.shape[0] - 1), axis=0), np.clip(x_coords.astype(int), 0, img.shape[1] - 1), axis=1)
                    if img.shape[0] != 28 or img.shape[1] != 28:
                        img = np.pad(img, ((0, max(0, 28 - img.shape[0])), (0, max(0, 28 - img.shape[1]))), mode='edge')
                        img = img[:28, :28]

            augmented[i, 0] = img

        return augmented