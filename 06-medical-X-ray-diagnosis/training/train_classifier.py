import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_classifier(model, train_loader, test_loader, config):
    """Train multi-label disease classifier"""
    
    device = config.DEVICE
    model = model.to(device)
    
    # Loss: Binary Cross Entropy for multi-label
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.CLASSIFIER_LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("=" * 60)
    print("🏥 TRAINING DISEASE CLASSIFIER")
    print("=" * 60)
    
    for epoch in range(config.CLASSIFIER_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLASSIFIER_EPOCHS}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions, _ = model(images)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predicted = (predictions > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                predictions, _ = model(images)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted = (predictions > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
        
        # Store history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.CLASSIFIER_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config.CHECKPOINT_DIR}/classifier_epoch_{epoch+1}.pth")
    
    return history