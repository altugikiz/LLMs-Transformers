"""
Multi-Head Attention implementation from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, n_heads, seq_len_q, d_k)
            key: (batch_size, n_heads, seq_len_k, d_k)
            value: (batch_size, n_heads, seq_len_v, d_k) where seq_len_v = seq_len_k
            mask: (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
            
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
                mask: torch.Tensor = None) -> torch.Tensor:
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
        
        # 1. Linear projections and split into heads
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_v, d_model)
        
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
        # Then transpose: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        # If mask is provided, adapt it for multi-head
        if mask is not None:
            # mask: (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
            
        context, attn_weights = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads and project
        # context: (batch_size, n_heads, seq_len_q, d_k)
        # -> (batch_size, seq_len_q, n_heads, d_k) -> (batch_size, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output


# Test the attention mechanism
if __name__ == "__main__":
    print("🧪 Testing Multi-Head Attention...")
    
    # Parameters
    batch_size = 2
    seq_len = 5
    d_model = 512
    n_heads = 8
    
    # Create random inputs
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Test ScaledDotProductAttention
    d_k = d_model // n_heads
    mha_attention = ScaledDotProductAttention()
    
    # Reshape for testing scaled attention
    Q_test = query.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    K_test = key.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    V_test = value.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    
    context, weights = mha_attention(Q_test, K_test, V_test)
    print(f"\nScaledDotProductAttention:")
    print(f"  Context shape: {context.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha(query, key, value)
    print(f"\nMultiHeadAttention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {output.norm().item():.4f}")
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask = torch.tril(mask)  # Causal mask for decoder
    output_masked = mha(query, key, value, mask)
    print(f"\nWith causal mask:")
    print(f"  Output shape: {output_masked.shape}")
    
    # Check parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n📊 MultiHeadAttention parameters: {total_params:,}")