import numpy as np
from engine.mha import MultiHeadAttention
from engine.layer_norm import LayerNormalization
from engine.ffn import FeedForwardNetwork

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x):
        # 1. Sublayer: Multi-Head Attention + Residual + LayerNorm
        attn_output, _ = self.mha.forward(x, x, x)
        x = self.norm1.forward(x + attn_output) # Residual connection
        
        # 2. Sublayer: Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output) # Residual connection
        
        return x