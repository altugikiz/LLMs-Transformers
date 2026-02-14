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
        
        # Farklı initialization dene
        # nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        
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
        eos_token_id: int = None,
        debug: bool = False
    ):
        """
        Generate text autoregressively.
        
        Args:
            prompt: Input prompt of shape (batch_size, prompt_len)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, only sample from top-p probability mass
            eos_token_id: Stop generation when this token is generated
            debug: Print debug information
        """
        self.eval()
        generated = prompt
        vocab_size = self.vocab_size
        
        with torch.no_grad():
            for step in range(max_length - prompt.size(1)):
                # Forward pass
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :].clone()  # Clone to avoid modifying original
                
                # DEBUG: İlk adımda logits'i göster
                if debug and step == 0:
                    print(f"\n🔍 Step {step} - Raw logits stats:")
                    print(f"  shape: {next_token_logits.shape}")
                    print(f"  min: {next_token_logits.min().item():.2f}")
                    print(f"  max: {next_token_logits.max().item():.2f}")
                    print(f"  mean: {next_token_logits.mean().item():.2f}")
                    print(f"  std: {next_token_logits.std().item():.2f}")
                    
                    # Raw probabilities (softmax öncesi)
                    raw_probs = F.softmax(next_token_logits, dim=-1)
                    top_raw_probs, top_raw_indices = torch.topk(raw_probs[0], 10)
                    print(f"\n  Top 10 tokens (raw softmax):")
                    for i in range(10):
                        print(f"    {top_raw_indices[i].item():3d}: {top_raw_probs[i].item():.4f}")
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    if top_k > vocab_size:
                        top_k = vocab_size
                        if debug and step == 0:
                            print(f"  ⚠️ top_k > vocab_size, setting to {vocab_size}")
                    
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    
                    # Create new tensor with -inf for all but top-k
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = filtered_logits
                    
                    if debug and step == 0:
                        print(f"\n  After top-k (k={top_k}):")
                        print(f"    min: {next_token_logits.min().item():.2f}")
                        print(f"    max: {next_token_logits.max().item():.2f}")
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Check if all probabilities are zero (shouldn't happen, but just in case)
                if probs.sum() == 0:
                    if debug:
                        print(f"  ⚠️ All probabilities are zero! Using uniform distribution.")
                    probs = torch.ones_like(probs) / vocab_size
                
                # DEBUG: İlk adımda sonuçları göster
                if debug and step == 0:
                    print(f"\n  Final probabilities:")
                    print(f"    sum: {probs.sum().item():.2f}")
                    print(f"    max: {probs.max().item():.4f}")
                    print(f"    min: {probs.min().item():.4f}")
                    print(f"    mean: {probs.mean().item():.4f}")
                    
                    # Top 5 final probabilities
                    top_probs, top_indices = torch.topk(probs[0], 10)
                    print(f"\n  Top 10 final tokens:")
                    for i in range(10):
                        print(f"    {top_indices[i].item():3d}: {top_probs[i].item():.4f}")
                    
                    # Check if the first token is being repeated
                    first_token = prompt[0, 0].item()
                    if first_token in top_indices:
                        first_token_prob = probs[0, first_token].item()
                        first_token_rank = (top_indices == first_token).nonzero(as_tuple=True)[0]
                        rank_str = first_token_rank[0].item() + 1 if len(first_token_rank) > 0 else "not in top 10"
                        print(f"\n  Prompt'un ilk token'ı ({first_token}) olasılığı: {first_token_prob:.4f}, sıralama: {rank_str}")
                
                # Sample from the distribution
                try:
                    next_token = torch.multinomial(probs, num_samples=1)
                except RuntimeError as e:
                    # If multinomial fails (shouldn't happen now), use argmax
                    if debug:
                        print(f"  ⚠️ Multinomial failed: {e}, using argmax")
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # DEBUG: Her adımda ne ürettiğimizi göster
                if debug:
                    print(f"\n  Step {step+1} - Generated token: {next_token[0,0].item()}")
                
                # Stop if EOS token is generated
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    if debug:
                        print(f"  🏁 EOS token generated, stopping")
                    break
        
        return generated


# Test the full model
if __name__ == "__main__":
    print("🧪 Testing complete Transformer model...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_seq_len = 50
    
    # Create model
    model = ColorTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Create random input
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    print(f"\n✅ Forward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    
    # Test generation with debug
    prompt = torch.randint(1, vocab_size, (1, 3))
    print(f"\n🔄 Testing generation with debug...")
    generated = model.generate(
        prompt,
        max_length=8,
        temperature=1.0,
        top_k=40,
        debug=True
    )
    print(f"\n✅ Generation successful!")
    print(f"  Generated shape: {generated.shape}")