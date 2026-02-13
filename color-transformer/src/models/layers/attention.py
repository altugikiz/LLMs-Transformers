"""
Multi-Head Attention implementation from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, n_heads, seq_len_q, d_k)
            key: (batch_size, n_heads, seq_len_k, d_k)
            value: (batch_size, n_heads, seq_len_v, d_k) where seq_len_v = seq_len_k
            mask: (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k)
            
        Returns:
            Context vector: (batch_size, n_heads, seq_len_q, d_k)
            Attention weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, value)
        
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Splits the model dimension into multiple heads for parallel attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model) where seq_len_v = seq_len_k
            mask: (batch_size, seq_len_q, seq_len_k) optional attention mask
            
        Returns:
            Output: (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 1. Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_v, d_model)
        
        # 2. Split into heads: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k)  # seq_len_v = seq_len_k
        
        # 3. Transpose: (batch_size, n_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4. Apply attention
        if mask is not None:
            # mask: (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
            
        context, attn_weights = self.attention(Q, K, V, mask)
        
        # 5. Transpose back and concatenate heads
        # context: (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        
        # 6. Combine heads: (batch_size, seq_len_q, d_model)
        context = context.view(batch_size, seq_len_q, self.d_model)
        
        # 7. Final linear projection
        output = self.w_o(context)
        
        return output


# Test the attention mechanism
if __name__ == "__main__":
    print("🧪 Testing Multi-Head Attention...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    
    print(f"\n📊 Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k: {d_model // n_heads}")
    
    # Create random inputs
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n📥 Input shapes:")
    print(f"  query: {query.shape}")
    print(f"  key: {key.shape}")
    print(f"  value: {value.shape}")
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha(query, key, value)
    
    print(f"\n📤 Output shape:")
    print(f"  output: {output.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {output.shape} vs {(batch_size, seq_len, d_model)}"
    
    # Test with different sequence lengths
    print(f"\n🔄 Testing with different sequence lengths...")
    query_short = torch.randn(batch_size, 5, d_model)
    key_long = torch.randn(batch_size, 15, d_model)
    value_long = torch.randn(batch_size, 15, d_model)
    
    output_diff = mha(query_short, key_long, value_long)
    print(f"  Query shape: {query_short.shape}")
    print(f"  Key/Value shape: {key_long.shape}")
    print(f"  Output shape: {output_diff.shape}")
    print(f"  Expected: ({batch_size}, 5, {d_model})")
    
    assert output_diff.shape == (batch_size, 5, d_model), f"Shape mismatch for different seq lengths"
    
    # Test with mask
    print(f"\n🎭 Testing with causal mask...")
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    output_masked = mha(query, key, value, mask)
    print(f"  Output shape with mask: {output_masked.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n📊 MultiHeadAttention parameters: {total_params:,}")
    
    print(f"\n✅ All tests passed!")