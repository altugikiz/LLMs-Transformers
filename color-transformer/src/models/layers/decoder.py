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
        print(f"\n  🔍 DecoderLayer Debug:")
        print(f"    Input x shape: {x.shape}, elements: {x.numel()}")
        
        # 1. Masked multi-head self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        print(f"    After attention - attn_output shape: {attn_output.shape}, elements: {attn_output.numel()}")
        
        # Residual connection + dropout + layer norm
        x = x + self.self_attn_dropout(attn_output)
        print(f"    After residual - x shape: {x.shape}, elements: {x.numel()}")
        
        x = self.self_attn_norm(x)
        print(f"    After norm - x shape: {x.shape}, elements: {x.numel()}")
        
        # 2. Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        print(f"    After FFN - ff_output shape: {ff_output.shape}, elements: {ff_output.numel()}")
        
        x = x + self.ff_dropout(ff_output)
        print(f"    After FFN residual - x shape: {x.shape}, elements: {x.numel()}")
        
        x = self.ff_norm(x)
        print(f"    Final output shape: {x.shape}, elements: {x.numel()}")
        
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
        print(f"\n🔍 Decoder Debug:")
        print(f"  Input to decoder: {x.shape}, elements: {x.numel()}")
        print(f"  Mask shape: {mask.shape if mask is not None else None}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n  Layer {i+1}:")
            x = layer(x, mask)
        
        x = self.final_norm(x)
        print(f"\n  Final decoder output: {x.shape}, elements: {x.numel()}")
        
        return x


# Test Decoder
if __name__ == "__main__":
    print("🧪 Testing Decoder layers...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    d_ff = 1024
    n_layers = 3
    
    print(f"\n📊 Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  n_layers: {n_layers}")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    
    print(f"\n📥 Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  mask: {mask.shape}")
    
    # Test single decoder layer
    print("\n" + "="*50)
    print("Testing single DecoderLayer:")
    print("="*50)
    
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
    layer_output = decoder_layer(x, mask)
    
    print(f"\n✅ Single layer output shape: {layer_output.shape}")
    assert layer_output.shape == x.shape, f"Shape mismatch: {layer_output.shape} vs {x.shape}"
    
    # Test full decoder stack
    print("\n" + "="*50)
    print(f"Testing full Decoder ({n_layers} layers):")
    print("="*50)
    
    decoder = Decoder(n_layers, d_model, n_heads, d_ff)
    decoder_output = decoder(x, mask)
    
    print(f"\n✅ Full decoder output shape: {decoder_output.shape}")
    assert decoder_output.shape == x.shape, f"Shape mismatch: {decoder_output.shape} vs {x.shape}"
    
    # Parameter count
    layer_params = sum(p.numel() for p in decoder_layer.parameters())
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Parameter counts:")
    print(f"  Single layer: {layer_params:,}")
    print(f"  Full decoder: {total_params:,}")
    
    print("\n✅ All decoder tests passed!")