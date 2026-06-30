import numpy as np

class Layer:
    """Base layer class"""
    def __init__(self):
        self.cache = {}
    
    def forward(self, X, training=True):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer with He initialization and isolated random generation"""
    def __init__(self, input_size, output_size, use_bias=True, rng=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        generator = rng if rng is not None else np.random.default_rng(42)
        
        self.W = generator.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, output_size))
        self.b = np.zeros((1, output_size)) if use_bias else None
        
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b) if use_bias else None
        self.v_b = np.zeros_like(self.b) if use_bias else None
    
    def forward(self, X, training=True):
        self.cache['X'] = X
        self.cache['n_samples'] = X.shape[0]  # Store batch size
        Z = np.dot(X, self.W)
        if self.use_bias:
            Z += self.b
        return Z
    
    def backward(self, grad):
        X = self.cache['X']
        n_samples = X.shape[0]  # Get batch size
        
        dW = np.dot(X.T, grad) / n_samples
        dX = np.dot(grad, self.W.T)
        
        if self.use_bias:
            db = np.sum(grad, axis=0, keepdims=True) / n_samples
        else:
            db = None
        
        return dX, dW, db
    
class Activation(Layer):
    """Activation layer handling forward and backward non-linear functions"""
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
    
    def forward(self, X, training=True):
        self.cache['X'] = X
        
        if self.activation_type == 'relu':
            return np.maximum(0, X)
        elif self.activation_type == 'leaky_relu':
            return np.where(X > 0, X, 0.01 * X)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation_type == 'softmax':
            X_stable = X - np.max(X, axis=1, keepdims=True)
            exp_X = np.exp(X_stable)
            return exp_X / np.sum(exp_X, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
    
    def backward(self, grad):
        X = self.cache['X']
        
        if self.activation_type == 'relu':
            return grad * (X > 0).astype(float)
        elif self.activation_type == 'leaky_relu':
            return grad * np.where(X > 0, 1.0, 0.01)
        elif self.activation_type == 'sigmoid':
            sig = 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
            return grad * sig * (1 - sig)
        elif self.activation_type == 'softmax':
            return grad  # Combined cleanly with cross-entropy loss gradients
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")

class BatchNorm(Layer):
    """Batch Normalization layer with running stats for validation inference"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)
    
    def forward(self, X, training=True):
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            X_norm = (X - mean) / np.sqrt(var + self.eps)
            self.cache = {'mean': mean, 'var': var, 'X_norm': X_norm}
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * X_norm + self.beta
    
    def backward(self, grad):
        var = self.cache['var']
        X_norm = self.cache['X_norm']
        n_samples = grad.shape[0]
        
        dgamma = np.sum(grad * X_norm, axis=0, keepdims=True) / n_samples
        dbeta = np.sum(grad, axis=0, keepdims=True) / n_samples
        
        grad_mean = np.mean(grad, axis=0, keepdims=True)
        grad_var = np.mean(grad * X_norm, axis=0, keepdims=True)
        dX = (self.gamma / np.sqrt(var + self.eps)) * (grad - grad_mean - X_norm * grad_var)
        
        return dX, dgamma, dbeta
    
class Dropout(Layer):
    """Dropout regularization layer utilizing a dedicated seed engine"""
    def __init__(self, rate=0.5, rng=None):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.rng = rng if rng is not None else np.random.default_rng(42)
    
    def forward(self, X, training=True):
        if training and self.rate > 0:
            self.mask = (self.rng.random(X.shape) > self.rate) / (1 - self.rate)
            return X * self.mask
        else:
            return X
    
    def backward(self, grad):
        if self.mask is not None:
            return grad * self.mask
        return grad