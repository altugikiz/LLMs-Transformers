"""
Transformer Decoder Layer.
"""

import torch
import torch.nn as nn
from src.models.layers.attention import MultiHeadAttention
from src.models.layers.feedforward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(p=dropout)
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Self-attention with residual
        attn_output = self.self_attention(x, x, x, mask)
        x = x + self.self_attn_dropout(attn_output)
        x = self.self_attn_norm(x)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = x + self.ff_dropout(ff_output)
        x = self.ff_norm(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.final_norm(x)