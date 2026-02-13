"""
Multi-Head Attention implementation from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.epsilon = 1e-8  # Küçük bir epsilon ekle
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        d_k = query.size(-1)
        
        # Güvenli bölme için d_k sıfır olmasın
        d_k = max(d_k, self.epsilon)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Debug: scores'u kontrol et
        if torch.isnan(scores).any():
            print("⚠️ NaN in attention scores before mask!")
        
        if mask is not None:
            # Mask uygularken -inf kullanma, çok büyük negatif sayı kullan
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Debug: mask sonrası kontrol
        if torch.isnan(scores).any():
            print("⚠️ NaN in attention scores after mask!")
        
        # Softmax uygula
        attn_weights = F.softmax(scores, dim=-1)
        
        # Debug: attention weights'i kontrol et
        if torch.isnan(attn_weights).any():
            print("⚠️ NaN in attention weights!")
        
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            mask = mask.unsqueeze(1)
        
        context, _ = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.w_o(context)
        
        return output