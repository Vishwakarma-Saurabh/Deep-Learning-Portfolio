import torch
import numpy as np
from collections import Counter
import re

class Vocabulary:
    """Simple vocabulary builder"""
    def __init__(self, max_size=25000):
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}
    
    def build(self, texts):
        """Build vocabulary from list of texts"""
        # Count all words
        for text in texts:
            words = tokenize_text(text)
            for word in words:
                self.word_count[word] = self.word_count.get(word, 0) + 1
        
        # Sort by frequency and add to vocabulary
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if len(self.word2idx) >= self.max_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
        
        print(f"📚 Vocabulary size: {len(self.word2idx)} words")
        return self
    
    def encode(self, text):
        """Convert text to token indices"""
        words = tokenize_text(text)
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
    
    def decode(self, indices):
        """Convert token indices back to text"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])


def tokenize_text(text):
    """Simple tokenizer"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split by whitespace
    return text.split()


def pad_sequence(sequence, max_length, pad_value=0):
    """Pad or truncate sequence to max_length"""
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        pad_len = max_length - len(sequence)
        return sequence + [pad_value] * pad_len


def build_vocab(train_texts, max_size=25000):
    """Build vocabulary from training texts"""
    vocab = Vocabulary(max_size)
    vocab.build(train_texts)
    return vocab