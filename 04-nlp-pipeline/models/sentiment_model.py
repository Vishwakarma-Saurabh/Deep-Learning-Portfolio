import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    """
    LSTM-based sentiment analysis model
    Architecture: Embedding → LSTM → Dropout → Dense → Output
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, sequence_length)
        """
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Take the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            # (batch_size, hidden_dim * 2)
        else:
            hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Output
        output = self.fc(hidden)  # (batch_size, 1)
        output = self.sigmoid(output)  # (batch_size, 1)
        
        return output.squeeze(1)  # (batch_size,)


class GRUSentimentModel(nn.Module):
    """
    GRU-based sentiment analysis model (faster than LSTM)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, hidden = self.gru(x)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        output = self.sigmoid(output)
        
        return output.squeeze(1)