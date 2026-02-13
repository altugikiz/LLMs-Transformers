"""
Transformer Decoder Layer.
Combines masked multi-head attention and feed-forward network.
"""

import torch
import torch.nn as nn
from src.models.layers.attention import MultiHeadAttention
from src.models.layers.feedforward import FeedForward


class DecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Masked Multi-Head Self-Attention
    2. Feed-Forward Network
    Each sub-layer has residual connection and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(p=dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Causal mask to prevent looking ahead (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # 1. Masked multi-head self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = x + self.self_attn_dropout(attn_output)
        x = self.self_attn_norm(x)
        
        # 2. Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.ff_dropout(ff_output)
        x = self.ff_norm(x)
        
        return x


class Decoder(nn.Module):
    """
    Stack of multiple decoder layers.
    """
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1):
        """
        Args:
            n_layers: Number of decoder layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Causal mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.final_norm(x)


# Test Decoder
if __name__ == "__main__":
    print("🧪 Testing Decoder layers...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Test single decoder layer
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
    layer_output = decoder_layer(x, mask)
    print(f"\nSingle DecoderLayer:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {layer_output.shape}")
    print(f"  Output norm: {layer_output.norm().item():.4f}")
    
    # Test full decoder stack
    decoder = Decoder(n_layers, d_model, n_heads, d_ff)
    decoder_output = decoder(x, mask)
    print(f"\nFull Decoder ({n_layers} layers):")
    print(f"  Output shape: {decoder_output.shape}")
    print(f"  Output norm: {decoder_output.norm().item():.4f}")
    
    # Parameter count
    layer_params = sum(p.numel() for p in decoder_layer.parameters())
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Parameter counts:")
    print(f"  Single layer: {layer_params:,}")
    print(f"  Full decoder: {total_params:,}")