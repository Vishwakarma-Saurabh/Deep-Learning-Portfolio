import numpy as np
from config import Config
from neural_network import NeuralNetwork
from utils import DataLoader, Visualizer
from train import Trainer

def main():
    # Set random seed for reproducibility
    Config.set_seed()
    
    # 1. Load and prepare data
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNIZER")
    print("=" * 60)
    
    data_loader = DataLoader()
    data_loader.load_mnist()
    
    # 2. Display sample images
    Visualizer.show_sample_images(data_loader.X_train, data_loader.y_train)
    
    # 3. Initialize neural network
    model = NeuralNetwork.from_config(Config)
    
    # Print network architecture
    print(f"Network Architecture:")
    print(f"  Input Layer: {Config.INPUT_SIZE} neurons")
    print(f"  Hidden Layer: {Config.HIDDEN_SIZE} neurons")
    print(f"  Output Layer: {Config.OUTPUT_SIZE} neurons")
    print(f"  Total Parameters: {model.W1.size + model.b1.size + model.W2.size + model.b2.size:,}")
    print()
    
    # 4. Train the model
    trainer = Trainer(model, data_loader)
    history = trainer.train(
        data_loader.X_train, data_loader.y_train,
        data_loader.X_val, data_loader.y_val,
        epochs=Config.EPOCHS
    )
    
    # 5. Plot training history
    Visualizer.plot_training_history(history)
    
    # 6. Evaluate on test set
    print("\nEvaluating on Test Set:")
    test_pred = model.predict(data_loader.X_test)
    test_accuracy = np.mean(test_pred == data_loader.y_test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # 7. Visualize learned weights
    Visualizer.plot_weights(model.W1, "Hidden Layer Weights")
    
    # 8. Show some predictions
    n_samples = 10
    indices = Config.rng.choice(len(data_loader.X_test), n_samples, replace=False)
    
    print("\nSample Predictions:")
    print("-" * 40)
    for i, idx in enumerate(indices):
        true_label = data_loader.y_test[idx]
        pred_label = test_pred[idx]
        status = "✓" if true_label == pred_label else "✗"
        print(f"Sample {i+1}: True: {true_label}, Predicted: {pred_label} {status}")

if __name__ == "__main__":
    main()