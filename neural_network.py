import numpy as np
from config import Config

class NeuralNetwork:
    def __init__(self, input_size=Config.INPUT_SIZE, hidden_size=Config.HIDDEN_SIZE, output_size=Config.OUTPUT_SIZE, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = rng if rng is not None else Config.rng
        
        # Weight Initialization (Xavier/Glorot)
        # W1: Input to Hidden
        self.W1 = self.rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: Hidden to Output
        self.W2 = self.rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store intermediate values for backpropagation
        self.cache = {}
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def softmax(self, Z):
        # Subtract max for numerical stability
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X, training=True, dropout_rate=0.2):
        # Layer 1: Input to Hidden
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # Dropout (only during training)
        if training:
            dropout_mask = (self.rng.random(A1.shape) > dropout_rate) / (1 - dropout_rate)
            A1 = A1 * dropout_mask
        else:
            dropout_mask = np.ones_like(A1)
        
        # Layer 2: Hidden to Output
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        
        # Store for backpropagation
        self.cache = {
            'X': X,              # Input
            'Z1': Z1,            # Pre-activation of layer 1
            'A1': A1,            # Post-activation of layer 1
            'Z2': Z2,            # Pre-activation of layer 2
            'A2': A2,            # Post-activation of layer 2 (output probabilities)
            'dropout_mask': dropout_mask
        }
        
        return A2
    
    def backward(self, y_one_hot, l2_lambda=0.001):
        # Extract cached values
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        
        n_samples = X.shape[0]
        
        # Output layer gradients (Softmax + Cross-entropy)
        # dL/dZ2 = A2 - y (simplified derivative)
        dZ2 = A2 - y_one_hot
        dW2 = np.dot(A1.T, dZ2) / n_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        # ReLU derivative
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        # Apply dropout mask to gradients
        dZ1 = dZ1 * self.cache['dropout_mask']
        
        dW1 = np.dot(X.T, dZ1) / n_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
        
        # L2 Regularization gradient (adds weight decay)
        if l2_lambda > 0:
            dW2 += (l2_lambda / n_samples) * self.W2
            dW1 += (l2_lambda / n_samples) * self.W1
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
    
    def update_params(self, gradients, learning_rate=0.01):
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
    
    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def compute_loss(self, y_true, y_pred, l2_lambda=0.001):
        n_samples = y_true.shape[0]
        
        # Cross-entropy loss
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        ce_loss = -np.sum(y_true * np.log(y_pred + eps)) / n_samples
        
        # L2 regularization loss
        l2_loss = (l2_lambda / (2 * n_samples)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        
        return ce_loss + l2_loss
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100