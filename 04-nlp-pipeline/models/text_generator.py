import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.Module):
    """
    LSTM-based text generation model
    Architecture: Embedding → LSTM → Dense → Softmax
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        x: (batch_size, sequence_length)
        hidden: (num_layers, batch_size, hidden_dim)
        """
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def generate(self, start_text, vocab, max_length=50, temperature=0.8):
        """
        Generate text from seed
        """
        self.eval()
        
        # Encode start text
        tokens = vocab.encode(start_text)
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        generated = tokens.copy()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                output, hidden = self(input_tensor, hidden)
                
                # Get last time step
                logits = output[0, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=0)
                
                # Sample from distribution
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to generated sequence
                generated.append(next_token)
                
                # Update input
                input_tensor = torch.tensor([[next_token]], dtype=torch.long)
                
                # Stop if EOS token
                if next_token == vocab.word2idx.get('<EOS>', 0):
                    break
        
        # Decode back to text
        return vocab.decode(generated)