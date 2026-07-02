import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from config import Config
from models.cnn import CNN
from data.dataset import DataLoader
from training.trainer import Trainer
from utils.visualization import Visualizer

def main():
    # Set seed
    Config.set_seed()
    
    # Load data
    data_loader = DataLoader()
    data_loader.set_limits(
        train=Config.MAX_TRAIN_SAMPLES,
        val=Config.MAX_VAL_SAMPLES,
        test=Config.MAX_TEST_SAMPLES,
    )
    data_loader.load_mnist(test_size=0.2, val_size=0.1)
    
    # Create model
    print("🧠 Creating CNN model...")
    model = CNN(Config, rng=Config.get_rng())
    
    # Print model summary
    total_params = sum(np.prod(p.shape) for p in model.params.values())
    print(f"📊 Total parameters: {total_params:,}")
    print()
    
    # Create trainer
    trainer = Trainer(model, data_loader, Config)
    
    # Train
    history = trainer.train(
        data_loader.X_train, data_loader.y_train,
        data_loader.X_val, data_loader.y_val
    )
    
    # Test
    trainer.test(data_loader.X_test, data_loader.y_test)
    
    # Visualize
    if Config.PLOT_TRAINING:
        visualizer = Visualizer()
        visualizer.plot_training_history(history)
        visualizer.plot_conv_weights(model)

if __name__ == "__main__":
    main()