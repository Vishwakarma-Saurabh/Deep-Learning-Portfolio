import os
import torch
import torch.nn as nn
from config import Config
from data import load_imdb, build_vocab, create_data_loaders
from models import SentimentModel, TextGenerator
from training import Trainer, evaluate_sentiment, generate_text
from utils import plot_training_history, predict_sentiment

def main():
    """Main entry point"""
    Config.set_seed()
    
    # ============ CREATE OUTPUT DIRECTORIES ============
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # ============ DATA ============
    print("\n" + "=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)
    
    # Load IMDB data
    data = load_imdb()
    
    # Build vocabulary
    vocab = build_vocab(data['train_texts'], max_size=Config.MAX_VOCAB_SIZE)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        data['train_texts'], data['train_labels'],
        data['test_texts'], data['test_labels'],
        vocab,
        batch_size=Config.BATCH_SIZE,
        max_length=Config.MAX_SEQUENCE_LENGTH
    )
    
    # ============ SENTIMENT MODEL ============
    print("\n" + "=" * 60)
    print("🧠 BUILDING SENTIMENT MODEL")
    print("=" * 60)
    
    sentiment_model = SentimentModel(
        vocab_size=len(vocab.word2idx),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL
    ).to(Config.DEVICE)
    
    print(f"Model: LSTM Sentiment Classifier")
    print(f"Parameters: {sum(p.numel() for p in sentiment_model.parameters()):,}")
    print(f"Device: {Config.DEVICE}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        sentiment_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Trainer
    trainer = Trainer(sentiment_model, optimizer, criterion, Config.DEVICE)
    
    # ============ TRAINING ============
    history = trainer.train(train_loader, test_loader, epochs=Config.EPOCHS)
    
    # ============ EVALUATION ============
    print("\n" + "=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)
    
    accuracy = evaluate_sentiment(sentiment_model, test_loader, Config.DEVICE)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # ============ VISUALIZATION ============
    plot_training_history(history, save_path='outputs/sentiment_history.png')
    
    # ============ PREDICT SENTIMENT ============
    print("\n" + "=" * 60)
    print("🔮 PREDICTING SENTIMENT")
    print("=" * 60)
    
    test_texts = [
        "I absolutely loved this movie, it was fantastic!",
        "This was the worst film I've ever seen, terrible!",
        "The acting was good but the plot was confusing."
    ]
    
    for text in test_texts:
        predict_sentiment(text, sentiment_model, vocab, Config.DEVICE)
    
    # ============ TEXT GENERATION ============
    print("\n" + "=" * 60)
    print("✍️ TEXT GENERATION")
    print("=" * 60)
    
    # Build text generator with same vocabulary
    generator = TextGenerator(
        vocab_size=len(vocab.word2idx),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Copy weights from sentiment model (for demo)
    generator.embedding.weight.data = sentiment_model.embedding.weight.data.clone()
    
    # Generate text
    seed_texts = [
        "I love",
        "The movie was",
        "Once upon a time"
    ]
    
    for seed in seed_texts:
        generate_text(
            generator,
            vocab,
            seed,
            max_length=Config.GENERATION_SEQUENCE_LENGTH,
            temperature=Config.TEMPERATURE
        )
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT 4 COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()