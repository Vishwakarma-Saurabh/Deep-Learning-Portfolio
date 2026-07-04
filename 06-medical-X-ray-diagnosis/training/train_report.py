import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_report_generator(model, train_loader, test_loader, vocab, config):
    """Train report generator"""
    
    device = config.DEVICE
    model = model.to(device)
    
    # Loss: Cross Entropy for next word prediction
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.PAD_TOKEN])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.REPORT_LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("=" * 60)
    print("📝 TRAINING REPORT GENERATOR")
    print("=" * 60)
    
    for epoch in range(config.REPORT_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.REPORT_EPOCHS}")
        for images, _, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions[:, :-1])
            
            # Compute loss
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                captions[:, 1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, _, captions in test_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.REPORT_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Generate sample reports
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_images = next(iter(test_loader))[0][:4].to(device)
                reports = model.generate_report(
                    sample_images,
                    vocab,
                    {v: k for k, v in vocab.items()},
                    max_length=50
                )
                print("\n📝 Sample Generated Reports:")
                for i, report in enumerate(reports):
                    print(f"  Report {i+1}: {report}")
                print()
    
    return history