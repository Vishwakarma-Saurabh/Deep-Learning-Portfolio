import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
import os
import pickle

class IMDBDataset(Dataset):
    """IMDB sentiment dataset"""
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        tokens = self.vocab.encode(text)
        # Pad/truncate
        tokens = pad_sequence(tokens, self.max_length, pad_value=0)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_imdb(sample_size=None):
    """
    Load IMDB dataset (using a subset for speed)
    """
    print("📂 Loading IMDB dataset...")
    
    # Try to load cached data
    cache_file = 'data/imdb_data.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            print(f"✅ Loaded from cache: {len(data['train_texts'])} training samples")
            return data
    
    # If no cache, create sample data
    # For demo, we'll use synthetic data (since real IMDB is large)
    # In practice, you'd load from torchtext or other sources
    
    print("📝 Generating sample data for demonstration...")
    
    # Sample positive reviews
    positive_reviews = [
        "I loved this movie it was amazing and thrilling from start to finish",
        "Great film with excellent acting and beautiful cinematography",
        "A wonderful story that kept me engaged throughout",
        "Fantastic movie highly recommended for everyone",
        "Brilliant performance by the entire cast",
        "An instant classic that will be remembered for years",
        "Captivating and thought-provoking from beginning to end",
        "One of the best films I have ever seen",
        "Superb direction and a compelling narrative",
        "A masterpiece of modern cinema",
        "Incredible visual effects and stunning visuals",
        "The acting was top-notch and the plot was engaging",
        "A heartwarming story that will make you smile",
        "Must watch for any film enthusiast",
        "Excellent pacing and great character development",
        "A truly inspiring and uplifting movie",
        "Gripping storyline with unexpected twists",
        "The soundtrack was phenomenal and added to the atmosphere",
        "Great chemistry between the leads",
        "An unforgettable cinematic experience"
    ]
    
    # Sample negative reviews
    negative_reviews = [
        "I hated this movie it was boring and predictable",
        "Terrible film with horrible acting and no plot",
        "A complete waste of time and money",
        "Awful movie that I regret watching",
        "The worst film I have ever seen",
        "Disappointing from start to finish",
        "Boring and unoriginal with no redeeming qualities",
        "A disaster of a movie",
        "Horrible script and terrible performances",
        "I would not recommend this to anyone",
        "Completely forgettable and bland",
        "The acting was wooden and the plot was confusing",
        "A tedious and uninspired mess",
        "Painfully dull with no substance",
        "Laughably bad and poorly made",
        "Frustrating and disappointing",
        "A total letdown with no payoff",
        "The worst thing I have seen this year",
        "Dreadful execution of a mediocre idea",
        "Utterly pointless and boring"
    ]
    
    # Create labels: 1 for positive, 0 for negative
    train_texts = positive_reviews * 25 + negative_reviews * 25  # 1000 samples
    train_labels = [1] * 500 + [0] * 500
    
    test_texts = positive_reviews[:10] * 5 + negative_reviews[:10] * 5  # 100 samples
    test_labels = [1] * 50 + [0] * 50
    
    data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }
    
    # Cache the data
    os.makedirs('data', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Generated {len(train_texts)} training samples")
    print(f"✅ Generated {len(test_texts)} test samples")
    
    return data


def create_data_loaders(train_texts, train_labels, test_texts, test_labels, 
                        vocab, batch_size=64, max_length=200):
    """Create PyTorch DataLoaders"""
    
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def pad_sequence(sequence, max_length, pad_value=0):
    """Pad or truncate sequence"""
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        pad_len = max_length - len(sequence)
        return sequence + [pad_value] * pad_len