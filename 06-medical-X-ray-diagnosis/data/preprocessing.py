import torch
import torchvision.transforms as transforms
import re
from collections import Counter
from config import Config

def preprocess_image(image):
    """Preprocess X-ray image for model input"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def tokenize_report(text):
    """Tokenize radiology report into words"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (keep periods for sentence structure)
    text = re.sub(r'[^\w\s.]', ' ', text)
    # Split into words
    tokens = text.split()
    return tokens


def build_vocab(reports, max_size=5000):
    """Build vocabulary from reports"""
    from config import Config
    
    word_counter = Counter()
    
    for report in reports:
        tokens = tokenize_report(report)
        word_counter.update(tokens)
    
    # Create vocabulary
    vocab = {
        Config.PAD_TOKEN: 0,
        Config.UNK_TOKEN: 1,
        Config.START_TOKEN: 2,
        Config.END_TOKEN: 3
    }
    
    # Add most common words
    for i, (word, _) in enumerate(word_counter.most_common(max_size - 4)):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Create reverse mapping
    idx_to_word = {v: k for k, v in vocab.items()}
    
    print(f"📚 Vocabulary size: {len(vocab)} words")
    return vocab, idx_to_word


def encode_report(report, vocab, max_length=100):
    """Convert report text to token indices"""
    from config import Config
    
    tokens = tokenize_report(report)
    
    # Add start token
    encoded = [vocab[Config.START_TOKEN]]
    
    # Encode tokens
    for token in tokens[:max_length-2]:
        encoded.append(vocab.get(token, vocab[Config.UNK_TOKEN]))
    
    # Add end token
    encoded.append(vocab[Config.END_TOKEN])
    
    # Pad to max length
    pad_len = max_length - len(encoded)
    if pad_len > 0:
        encoded += [vocab[Config.PAD_TOKEN]] * pad_len
    else:
        encoded = encoded[:max_length]
    
    return torch.tensor(encoded, dtype=torch.long)