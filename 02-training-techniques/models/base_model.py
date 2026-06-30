import numpy as np
from models.layers import Dense, Activation, BatchNorm, Dropout

class NeuralNetwork:
    """Enhanced neural network with modern training techniques and absolute reproducibility"""
    def __init__(self, input_size, hidden_sizes, output_size, 
                 dropout_rate=0.3, use_batch_norm=True, rng=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Save our isolated seed engine (passed down from config.py)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        self.layers = []
        self.params = {}
        self.grads = {}
        self.cache = {}
        
        self._build_model()
    
    def _build_model(self):
        """Build the network architecture passing the secure rng engine"""
        prev_size = self.input_size
        
        # Hidden layers
        for i, size in enumerate(self.hidden_sizes):
            # Dense layer with passed isolated random generator
            dense = Dense(prev_size, size, rng=self.rng)
            self.layers.append(dense)
            
            # Batch normalization
            if self.use_batch_norm:
                bn = BatchNorm(size)
                self.layers.append(bn)
            
            # Activation
            activation = Activation('relu')
            self.layers.append(activation)
            
            # Dropout with passed isolated random generator
            if self.dropout_rate > 0:
                dropout = Dropout(self.dropout_rate, rng=self.rng)
                self.layers.append(dropout)
            
            prev_size = size
        
        # Output layer
        dense_out = Dense(prev_size, self.output_size, rng=self.rng)
        self.layers.append(dense_out)
        activation_out = Activation('softmax')
        self.layers.append(activation_out)
        
        # Initialize parameter dictionary for optimizer
        self._init_params()
    
    def _init_params(self):
        """Initialize parameter dictionary for optimizer tracking"""
        self.params = {}
        self.grads = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                self.params[f'W{i}'] = layer.W
                self.params[f'b{i}'] = layer.b
                self.grads[f'dW{i}'] = np.zeros_like(layer.W)
                self.grads[f'db{i}'] = np.zeros_like(layer.b)
            
            elif isinstance(layer, BatchNorm):
                self.params[f'gamma{i}'] = layer.gamma
                self.params[f'beta{i}'] = layer.beta
                self.grads[f'dgamma{i}'] = np.zeros_like(layer.gamma)
                self.grads[f'dbeta{i}'] = np.zeros_like(layer.beta)
    
    def forward(self, X, training=True):
        """Forward propagation tracking output layers sequentially"""
        self.cache = {'X': X}
        current = X
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Dense, BatchNorm, Dropout)):
                current = layer.forward(current, training)
            else:  # Activation
                current = layer.forward(current)
            
            # Store everything using clean indices
            self.cache[f'layer_{i}'] = current
        
        return current
    
    def backward(self, y_one_hot, l2_lambda=0.0):
        """Backward propagation incorporating fixed cache indexing and L2 gradients"""
        n_samples = y_one_hot.shape[0]
        
        # FIX: Dynamically point directly to the final layer's cache position
        final_layer_idx = len(self.layers) - 1
        grad = self.cache[f'layer_{final_layer_idx}'] - y_one_hot
        
        self.grads = {}
        
        # Backward through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            if isinstance(layer, Activation):
                grad = layer.backward(grad)
            
            elif isinstance(layer, Dense):
                grad, dW, db = layer.backward(grad)
                
                # FIX: Add L2 regularized derivative penalty directly into the accumulated gradient
                if l2_lambda > 0:
                    dW += (l2_lambda / n_samples) * layer.W
                    
                self.grads[f'dW{i}'] = dW
                self.grads[f'db{i}'] = db
            
            elif isinstance(layer, BatchNorm):
                grad, dgamma, dbeta = layer.backward(grad)
                self.grads[f'dgamma{i}'] = dgamma
                self.grads[f'dbeta{i}'] = dbeta
            
            elif isinstance(layer, Dropout):
                grad = layer.backward(grad)
        
        return self.grads
    
    def update_params(self, optimizer):
        """Update parameters using optimizer configurations"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                self.params[f'W{i}'] = layer.W
                self.params[f'b{i}'] = layer.b
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                self.params[f'gamma{i}'] = layer.gamma
                self.params[f'beta{i}'] = layer.beta
        
        # Map gradient keys (e.g. dW0 -> W0) to match parameter names
        optimizer_grads = {
            key[1:]: value for key, value in self.grads.items()
            if key.startswith('d') and key[1:] in self.params
        }
        
        # Apply optimization update step
        self.params = optimizer.step(self.params, optimizer_grads)
        
        # Write back updated values to the individual structural layer classes
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.W = self.params[f'W{i}']
                layer.b = self.params[f'b{i}']
            elif isinstance(layer, BatchNorm):
                layer.gamma = self.params[f'gamma{i}']
                layer.beta = self.params[f'beta{i}']
    
    def predict(self, X):
        """Make class predictions"""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def compute_loss(self, y_true, y_pred, l2_lambda=0.0, label_smoothing=0.0):
        """Compute loss incorporating label smoothing properties and L2 validation tracking"""
        n_samples = y_true.shape[0]
        eps = 1e-8
        
        if label_smoothing > 0:
            y_true = (1 - label_smoothing) * y_true + label_smoothing / self.output_size
        
        ce_loss = -np.sum(y_true * np.log(y_pred + eps)) / n_samples
        
        l2_loss = 0
        if l2_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    l2_loss += np.sum(layer.W ** 2)
            l2_loss = (l2_lambda / (2 * n_samples)) * l2_loss
        
        return ce_loss + l2_loss
    
    def accuracy(self, X, y):
        """Calculate overall accuracy percentage"""
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100
    
    def get_parameters(self):
        """Get all trainable parameters for checkpointing"""
        params = {}
        for key, value in self.params.items():
            params[key] = value.copy()
        return params
    
    def set_parameters(self, params):
        """Set all trainable parameters from checkpoint"""
        for key, value in params.items():
            self.params[key] = value.copy()
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.W = self.params[f'W{i}']
                layer.b = self.params[f'b{i}']
            elif isinstance(layer, BatchNorm):
                layer.gamma = self.params[f'gamma{i}']
                layer.beta = self.params[f'beta{i}']