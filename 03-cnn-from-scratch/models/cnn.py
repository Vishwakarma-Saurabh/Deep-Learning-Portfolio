import numpy as np
from layers import Conv2D, MaxPool2D, AveragePool2D, Flatten, Dense, Activation, BatchNorm2D, Dropout

class CNN:
    """
    Convolutional Neural Network from scratch
    
    Architecture:
    Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → Dropout → Dense
    """
    def __init__(self, config, rng=None):
        self.config = config
        self.rng = rng if rng is not None else config.get_rng()
        
        self.layers = []
        self.params = {}
        self.grads = {}
        self.cache = {}
        
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture"""
        # Convert input channels
        in_channels = self.config.INPUT_SHAPE[0]
        
        # Convolutional layers
        for conv_config in self.config.CONV_LAYERS:
            conv = Conv2D(
                in_channels=conv_config['in_channels'],
                out_channels=conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                padding=conv_config['padding'],
                stride=conv_config['stride'],
                rng=self.rng
            )
            self.layers.append(conv)
            
            # Batch Normalization
            if self.config.USE_BATCH_NORM:
                bn = BatchNorm2D(conv_config['out_channels'])
                self.layers.append(bn)
            
            # Activation (ReLU)
            relu = Activation('relu')
            self.layers.append(relu)
            
            # Pooling
            pool_idx = len(self.config.POOL_LAYERS) - len(self.config.CONV_LAYERS) + \
                      self.config.CONV_LAYERS.index(conv_config)
            if pool_idx < len(self.config.POOL_LAYERS):
                pool = MaxPool2D(
                    pool_size=self.config.POOL_LAYERS[pool_idx]['pool_size'],
                    stride=self.config.POOL_LAYERS[pool_idx]['stride']
                )
                self.layers.append(pool)
            
            in_channels = conv_config['out_channels']
        
        # Flatten
        flatten = Flatten()
        self.layers.append(flatten)
        
        # Fully connected layers
        # Calculate flattened size
        # After conv layers: (batch, channels, 7, 7) for MNIST
        fc_input_size = 64 * 7 * 7  # Hardcoded for now, could compute dynamically
        
        # First FC layer
        fc1 = Dense(fc_input_size, self.config.FC_LAYERS[0], rng=self.rng)
        self.layers.append(fc1)
        
        # Activation
        relu_fc = Activation('relu')
        self.layers.append(relu_fc)
        
        # Dropout
        if self.config.DROPOUT_RATE > 0:
            dropout = Dropout(self.config.DROPOUT_RATE, rng=self.rng)
            self.layers.append(dropout)
        
        # Output layer
        fc2 = Dense(self.config.FC_LAYERS[0], self.config.NUM_CLASSES, rng=self.rng)
        self.layers.append(fc2)
        
        # Softmax
        softmax = Activation('softmax')
        self.layers.append(softmax)
        
        # Initialize parameters
        self._init_params()
    
    def _init_params(self):
        """Initialize parameter dictionary for optimizer"""
        self.params = {}
        self.grads = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                self.params[f'kernels_{i}'] = layer.kernels
                self.params[f'bias_{i}'] = layer.bias
                self.grads[f'dkernels_{i}'] = np.zeros_like(layer.kernels)
                self.grads[f'dbias_{i}'] = np.zeros_like(layer.bias)
            
            elif isinstance(layer, BatchNorm2D):
                self.params[f'gamma_{i}'] = layer.gamma
                self.params[f'beta_{i}'] = layer.beta
                self.grads[f'dgamma_{i}'] = np.zeros_like(layer.gamma)
                self.grads[f'dbeta_{i}'] = np.zeros_like(layer.beta)
            
            elif isinstance(layer, Dense):
                self.params[f'W_{i}'] = layer.W
                self.params[f'b_{i}'] = layer.b
                self.grads[f'dW_{i}'] = np.zeros_like(layer.W)
                self.grads[f'db_{i}'] = np.zeros_like(layer.b)
    
    def forward(self, X, training=True):
        """Forward propagation through all layers"""
        self.cache = {'X': X}
        current = X
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current, training)
            self.cache[f'layer_{i}'] = current
        
        return current
    
    def backward(self, y_one_hot, l2_lambda=0.0):
        """Backward propagation through all layers"""
        n_samples = y_one_hot.shape[0]
        
        # Start with loss gradient (combined with softmax)
        final_layer_idx = len(self.layers) - 1
        grad = self.cache[f'layer_{final_layer_idx}'] - y_one_hot
        
        self.grads = {}
        
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            if isinstance(layer, Activation):
                grad = layer.backward(grad)
            
            elif isinstance(layer, Conv2D):
                grad, dK, db = layer.backward(grad)
                
                if l2_lambda > 0:
                    dK += (l2_lambda) * layer.kernels
                
                self.grads[f'dkernels_{i}'] = dK
                self.grads[f'dbias_{i}'] = db
            
            elif isinstance(layer, BatchNorm2D):
                grad, dgamma, dbeta = layer.backward(grad)
                self.grads[f'dgamma_{i}'] = dgamma
                self.grads[f'dbeta_{i}'] = dbeta
            
            elif isinstance(layer, MaxPool2D) or isinstance(layer, AveragePool2D):
                grad = layer.backward(grad)
            
            elif isinstance(layer, Flatten):
                grad = layer.backward(grad)
            
            elif isinstance(layer, Dense):
                grad, dW, db = layer.backward(grad)
                
                if l2_lambda > 0:
                    dW += (l2_lambda) * layer.W
                
                self.grads[f'dW_{i}'] = dW
                self.grads[f'db_{i}'] = db
            
            elif isinstance(layer, Dropout):
                grad = layer.backward(grad)
        
        return self.grads
    
    def update_params(self, optimizer):
        """Update parameters using optimizer"""
        # Map gradients to parameter keys
        mapped_grads = {}
        for key in self.grads:
            # Remove 'd' prefix
            param_key = key[1:] if key.startswith('d') else key
            mapped_grads[param_key] = self.grads[key]
        
        # Update parameters
        self.params = optimizer.step(self.params, mapped_grads)
        
        # Write back to layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                layer.kernels = self.params[f'kernels_{i}']
                layer.bias = self.params[f'bias_{i}']
            
            elif isinstance(layer, BatchNorm2D):
                layer.gamma = self.params[f'gamma_{i}']
                layer.beta = self.params[f'beta_{i}']
            
            elif isinstance(layer, Dense):
                layer.W = self.params[f'W_{i}']
                layer.b = self.params[f'b_{i}']
    
    def get_parameters(self):
        return {key: value.copy() for key, value in self.params.items()}

    def set_parameters(self, params):
        self.params = {key: value.copy() for key, value in params.items()}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                layer.kernels = self.params[f'kernels_{i}']
                layer.bias = self.params[f'bias_{i}']
            elif isinstance(layer, BatchNorm2D):
                layer.gamma = self.params[f'gamma_{i}']
                layer.beta = self.params[f'beta_{i}']
            elif isinstance(layer, Dense):
                layer.W = self.params[f'W_{i}']
                layer.b = self.params[f'b_{i}']

    def forward_batches(self, X, batch_size=None, training=False):
        batch_size = batch_size or self.config.BATCH_SIZE
        if X.shape[0] <= batch_size:
            return self.forward(X, training=training)

        outputs = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            outputs.append(self.forward(X[start:end], training=training))
        return np.vstack(outputs)
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward_batches(X, training=False)
        return np.argmax(output, axis=1)
    
    def compute_loss(self, y_true, y_pred, l2_lambda=0.0, label_smoothing=0.0):
        """Compute loss with label smoothing"""
        n_samples = y_true.shape[0]
        eps = 1e-8
        
        if label_smoothing > 0:
            y_true = (1 - label_smoothing) * y_true + label_smoothing / self.config.NUM_CLASSES
        
        ce_loss = -np.sum(y_true * np.log(y_pred + eps)) / n_samples
        
        l2_loss = 0
        if l2_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, Conv2D):
                    l2_loss += np.sum(layer.kernels ** 2)
                elif isinstance(layer, Dense):
                    l2_loss += np.sum(layer.W ** 2)
            l2_loss = (l2_lambda / 2) * l2_loss
        
        return ce_loss + l2_loss
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100