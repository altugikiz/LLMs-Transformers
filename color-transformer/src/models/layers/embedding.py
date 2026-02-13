"""
Token and positional embeddings for Transformer.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token indices to dense vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
            Scaled by sqrt(d_model) as in "Attention is All You Need"
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings from the original Transformer paper.
    No learned parameters, uses fixed sin/cos functions.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for each dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input + positional embeddings, shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    """
    Combined token and positional embedding layer.
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len, dropout)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens with positional info, shape (batch_size, seq_len, d_model)
        """
        token_emb = self.token_embedding(x)
        return self.positional_embedding(token_emb)


# Test the embedding layers
if __name__ == "__main__":
    print("🧪 Testing embedding layers...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    d_model = 512
    max_seq_len = 50
    
    # Create random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test TokenEmbedding
    token_emb = TokenEmbedding(vocab_size, d_model)
    token_out = token_emb(x)
    print(f"\nTokenEmbedding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {token_out.shape}")
    print(f"  Output norm: {token_out.norm().item():.4f}")
    
    # Test PositionalEmbedding
    pos_emb = PositionalEmbedding(d_model, max_seq_len)
    pos_out = pos_emb(token_out)
    print(f"\nPositionalEmbedding:")
    print(f"  Output shape: {pos_out.shape}")
    print(f"  Output norm: {pos_out.norm().item():.4f}")
    
    # Test combined EmbeddingLayer
    emb_layer = EmbeddingLayer(vocab_size, d_model, max_seq_len)
    combined_out = emb_layer(x)
    print(f"\nCombined EmbeddingLayer:")
    print(f"  Output shape: {combined_out.shape}")
    print(f"  Output norm: {combined_out.norm().item():.4f}")
    
    # Visualize positional encodings (optional)
    import matplotlib.pyplot as plt
    
    # Get positional encodings for first 50 positions
    pe = pos_emb.pe[0, :50, :4].numpy()  # First 4 dimensions
    
    plt.figure(figsize=(12, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.plot(pe[:, i])
        plt.title(f'PE dim {i}')
        plt.xlabel('Position')
    plt.tight_layout()
    plt.savefig('positional_encodings.png')
    print("\n📊 Positional encoding plot saved to 'positional_encodings.png'")