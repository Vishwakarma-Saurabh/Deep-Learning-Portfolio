import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.input_shape = None
    
    def load_mnist(self, test_size=0.2, val_size=0.1):
        """Load MNIST dataset with proper shape (channels, height, width)."""
        print("📂 Loading MNIST dataset...")
        
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # Normalize and reshape to (channels, height, width)
        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, 1, 28, 28)  # (batch, channels, height, width)
        y = y.astype(np.int32)
        
        self.input_shape = X.shape[1:]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        if hasattr(self, 'max_train_samples') and self.max_train_samples is not None:
            X_train = X_train[:self.max_train_samples]
            y_train = y_train[:self.max_train_samples]
        if hasattr(self, 'max_val_samples') and self.max_val_samples is not None:
            X_val = X_val[:self.max_val_samples]
            y_val = y_val[:self.max_val_samples]
        if hasattr(self, 'max_test_samples') and self.max_test_samples is not None:
            X_test = X_test[:self.max_test_samples]
            y_test = y_test[:self.max_test_samples]

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Validation set: {X_val.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        return self
    
    def set_limits(self, train=None, val=None, test=None):
        self.max_train_samples = train
        self.max_val_samples = val
        self.max_test_samples = test
        return self

    def one_hot_encode(self, y, num_classes=10):
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, num_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot