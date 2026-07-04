import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Attention(nn.Module):
    """Attention mechanism for focusing on image regions"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(self, features, hidden_state):
        # features: (batch, 7, 7, 1024)
        # hidden_state: (batch, hidden_dim)
        
        batch_size, h, w, feat_dim = features.shape
        
        # Reshape features
        features_flat = features.view(batch_size, h * w, feat_dim)
        
        # Expand hidden state
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, h * w, -1)
        
        # Combine features and hidden state
        combined = torch.cat([features_flat, hidden_expanded], dim=2)
        
        # Compute attention scores
        scores = self.V(torch.tanh(self.W(combined))).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context = torch.sum(attention_weights.unsqueeze(-1) * features_flat, dim=1)
        
        return context, attention_weights


class ReportDecoder(nn.Module):
    """LSTM Decoder for generating radiology reports"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, feature_dim):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embed_dim + feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Attention
        self.attention = Attention(feature_dim, hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim + feature_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features, captions, training=True):
        """
        Args:
            features: (batch, 7, 7, 1024)
            captions: (batch, max_length)
        """
        batch_size = features.size(0)
        
        # Embed captions
        embeddings = self.embedding(captions)  # (batch, max_len, embed_dim)
        
        # Initialize LSTM
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(features.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(features.device)
        
        outputs = []
        hidden_state = None
        
        for t in range(embeddings.size(1)):
            # Get current word embedding
            word_embed = embeddings[:, t, :]  # (batch, embed_dim)
            
            # Get attention context
            if hidden_state is None:
                hidden_state = h0.squeeze(0)
            context, attention = self.attention(features, hidden_state)
            
            # Combine word embedding with context
            lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h0, c0))
            
            # Update hidden state
            hidden_state = h.squeeze(0)
            
            # Combine LSTM output with context
            combined = torch.cat([lstm_out.squeeze(1), context], dim=1)
            
            # Predict next word
            output = self.fc(self.dropout(combined))
            outputs.append(output)
            
            # Update for next step
            h0, c0 = h, c
        
        outputs = torch.stack(outputs, dim=1)  # (batch, max_len, vocab_size)
        return outputs
    
    def generate(self, features, vocab, idx_to_word, max_length=100, temperature=0.8):
        """Generate report from image features"""
        
        batch_size = features.size(0)
        
        # Initialize LSTM
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(features.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(features.device)
        
        # Start with <SOS> token
        start_token = vocab[Config.START_TOKEN]
        current_word = torch.tensor([start_token] * batch_size).to(features.device)
        
        generated = []
        hidden_state = None
        
        for _ in range(max_length):
            # Embed current word
            word_embed = self.embedding(current_word)  # (batch, embed_dim)
            
            # Get attention context
            if hidden_state is None:
                hidden_state = h0.squeeze(0)
            context, attention = self.attention(features, hidden_state)
            
            # Combine word embedding with context
            lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h0, c0))
            
            # Update hidden state
            hidden_state = h.squeeze(0)
            
            # Combine LSTM output with context
            combined = torch.cat([lstm_out.squeeze(1), context], dim=1)
            
            # Predict next word
            output = self.fc(combined)
            
            # Apply temperature
            probs = F.softmax(output / temperature, dim=1)
            
            # Sample next word
            next_word = torch.multinomial(probs, 1).squeeze(1)
            
            # Store generated word
            generated.append(next_word)
            
            # Check for end token
            if (next_word == vocab[Config.END_TOKEN]).all():
                break
            
            # Update for next step
            current_word = next_word
            h0, c0 = h, c
        
        # Decode tokens to text
        generated = torch.stack(generated, dim=1)
        reports = []
        
        for i in range(batch_size):
            tokens = []
            for t in range(generated.size(1)):
                token = generated[i, t].item()
                word = idx_to_word.get(token, Config.UNK_TOKEN)
                if word == Config.END_TOKEN:
                    break
                if word not in [Config.PAD_TOKEN, Config.START_TOKEN]:
                    tokens.append(word)
            reports.append(' '.join(tokens))
        
        return reports