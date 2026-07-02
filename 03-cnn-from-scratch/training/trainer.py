import numpy as np
import time
import os
import pickle
from data.dataset import DataLoader
from data.augmentations import DataAugmentation
from optimizers import get_optimizer

class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # Initialize optimizer
        self.optimizer = get_optimizer(
            config.OPTIMIZER,
            learning_rate=config.LEARNING_RATE,
            weight_decay=0.0,
            beta1=config.BETA1,
            beta2=config.BETA2,
            epsilon=config.EPSILON
        )
        
        # Initialize augmentations
        self.augmenter = DataAugmentation(
            rotation_range=10,
            shift_range=0.1,
            zoom_range=0.1,
            rng=config.get_rng()
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'time': []
        }
        
        self.best_val_acc = 0
        self.best_model_params = None
        self.patience_counter = 0
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    def train_epoch(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        batch_size = self.config.BATCH_SIZE
        
        y_train_one_hot = self.data_loader.one_hot_encode(y_train)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_one_hot[indices]
        
        total_batches = int(np.ceil(n_samples / batch_size))
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Augmentation
            if getattr(self.config, 'USE_AUGMENTATION', False):
                X_batch = self.augmenter(X_batch)
            
            # Forward
            y_pred = self.model.forward(X_batch, training=True)
            
            # Backward
            gradients = self.model.backward(y_batch, l2_lambda=self.config.WEIGHT_DECAY)
            
            # Update
            self.model.update_params(self.optimizer)

            if batch_idx % max(1, total_batches // 5) == 0 or batch_idx + 1 == total_batches:
                print(f"  batch {batch_idx + 1}/{total_batches} complete")
        
        # Evaluate
        y_pred_train = self.model.forward_batches(X_train, training=False)
        y_pred_val = self.model.forward_batches(X_val, training=False)
        
        train_loss = self.model.compute_loss(
            self.data_loader.one_hot_encode(y_train),
            y_pred_train,
            l2_lambda=self.config.WEIGHT_DECAY,
            label_smoothing=self.config.LABEL_SMOOTHING
        )
        val_loss = self.model.compute_loss(
            self.data_loader.one_hot_encode(y_val),
            y_pred_val,
            l2_lambda=self.config.WEIGHT_DECAY,
            label_smoothing=self.config.LABEL_SMOOTHING
        )
        
        train_acc = np.mean(np.argmax(y_pred_train, axis=1) == y_train) * 100
        val_acc = np.mean(np.argmax(y_pred_val, axis=1) == y_val) * 100
        
        return train_loss, val_loss, train_acc, val_acc
    
    def train(self, X_train, y_train, X_val, y_val):
        print("=" * 60)
        print("🚀 CNN TRAINING STARTED")
        print("=" * 60)
        print(f"Architecture: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Dense")
        print(f"Optimizer: {self.config.OPTIMIZER}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Dropout: {self.config.DROPOUT_RATE}")
        print(f"BatchNorm: {self.config.USE_BATCH_NORM}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.EPOCHS + 1):
            epoch_start = time.time()
            
            train_loss, val_loss, train_acc, val_acc = self.train_epoch(
                X_train, y_train, X_val, y_val
            )
            
            epoch_time = time.time() - epoch_start
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.config.LEARNING_RATE)
            self.history['time'].append(epoch_time)
            
            # Checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = self.model.get_parameters()
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    f'best_cnn_epoch_{epoch}.pkl'
                )
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'epoch': epoch,
                        'model_params': self.model.get_parameters(),
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }, f)
            else:
                self.patience_counter += 1
            
            # Print progress
            if epoch % self.config.PRINT_EVERY == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.config.EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Acc: {val_acc:.2f}% | "
                      f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if self.config.EARLY_STOPPING and self.patience_counter >= self.config.PATIENCE:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✅ Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        # Restore best model
        if self.best_model_params:
            self.model.set_parameters(self.best_model_params)
        
        return self.history
    
    def test(self, X_test, y_test):
        y_pred = self.model.forward_batches(X_test, training=False)
        test_loss = self.model.compute_loss(
            self.data_loader.one_hot_encode(y_test),
            y_pred,
            l2_lambda=0,
            label_smoothing=0
        )
        test_acc = self.model.accuracy(X_test, y_test)
        
        print(f"\n📊 Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        return test_loss, test_acc