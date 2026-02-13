"""
Complete Transformer model for color-attribute sentence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.embedding import EmbeddingLayer
from src.models.layers.decoder import Decoder


class ColorTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_len, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        
        self._init_weights()
        
        print(f"✅ Transformer model created:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")
        print(f"   Total parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
        causal_mask = causal_mask.unsqueeze(0)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1)
            mask = causal_mask * padding_mask.unsqueeze(-1)
        else:
            mask = causal_mask.repeat(batch_size, 1, 1)
        
        # Forward pass
        x = self.embedding(input_ids)
        x = self.decoder(x, mask)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None
    ):
        self.eval()
        generated = prompt
        
        with torch.no_grad():
            for _ in range(max_length - prompt.size(1)):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
        
        return generated