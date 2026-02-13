"""
Complete Transformer model for color-attribute sentence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.embedding import EmbeddingLayer
from src.models.layers.decoder import Decoder


class ColorTransformer(nn.Module):
    """
    Transformer model for generating color-attribute sentences.
    Uses only decoder (like GPT) for autoregressive generation.
    """
    
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
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Token ID for padding
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_len, dropout)
        
        # Decoder stack
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ Transformer model created:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")
        print(f"   Total parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Mask for padding of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        print(f"\n🔍 Transformer Forward Debug:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        
        batch_size, seq_len = input_ids.shape
        print(f"  batch_size: {batch_size}, seq_len: {seq_len}")
        
        # Create causal mask for autoregressive generation
        print(f"\n  Creating causal mask...")
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
        print(f"    causal_mask shape: {causal_mask.shape}")
        
        # Add batch and head dimensions: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask[None, None, :, :]
        print(f"    causal_mask after unsqueeze: {causal_mask.shape}")
        
        # Repeat for batch
        causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
        print(f"    causal_mask after repeat: {causal_mask.shape}, elements: {causal_mask.numel()}")
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            print(f"\n  Combining with padding mask...")
            # Convert attention mask to shape (batch_size, 1, 1, seq_len)
            padding_mask = attention_mask[:, None, None, :]
            print(f"    padding_mask shape: {padding_mask.shape}")
            
            # Expand padding mask to match causal mask dimensions
            padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
            print(f"    padding_mask after expand: {padding_mask.shape}")
            
            mask = causal_mask * padding_mask
            print(f"    final mask shape: {mask.shape}, elements: {mask.numel()}")
        else:
            mask = causal_mask
            print(f"\n  Using only causal mask: {mask.shape}")
        
        # Embed input
        print(f"\n  Embedding input...")
        x = self.embedding(input_ids)
        print(f"    embedded shape: {x.shape}, elements: {x.numel()}")
        
        # Pass through decoder
        print(f"\n  Passing through decoder...")
        x = self.decoder(x, mask)
        print(f"    decoder output shape: {x.shape}, elements: {x.numel()}")
        
        # Project to vocabulary
        print(f"\n  Projecting to vocabulary...")
        logits = self.output_projection(x)
        print(f"    logits shape: {logits.shape}, elements: {logits.numel()}")
        
        return logits
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Input prompt of shape (batch_size, prompt_len)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, only sample from top-p probability mass
            eos_token_id: Stop generation when this token is generated
            
        Returns:
            Generated token IDs of shape (batch_size, generated_len)
        """
        self.eval()
        batch_size = prompt.size(0)
        generated = prompt
        
        with torch.no_grad():
            for step in range(max_length - prompt.size(1)):
                print(f"\n  Generation step {step+1}")
                
                # Forward pass
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
        
        return generated


# Test the full model
if __name__ == "__main__":
    print("🧪 Testing complete Transformer model...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    d_model = 256
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    max_seq_len = 50
    
    print(f"\n📊 Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_layers: {n_layers}")
    print(f"  d_ff: {d_ff}")
    
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
    
    print(f"\n📥 Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # Forward pass
    print("\n" + "="*50)
    print("Testing forward pass:")
    print("="*50)
    
    logits = model(input_ids, attention_mask)
    
    print(f"\n✅ Forward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing generation:")
    print("="*50)
    
    prompt = torch.randint(1, vocab_size, (1, 3))  # Single sample with 3 tokens
    generated = model.generate(
        prompt,
        max_length=10,
        temperature=0.8,
        top_k=40,
        eos_token_id=3  # Assuming <EOS> is token 3
    )
    print(f"\n✅ Generation successful!")
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated sequence: {generated[0].tolist()}")
    
    # Memory usage
    param_count = model.count_parameters()
    param_size = param_count * 4 / (1024 ** 2)  # Assuming float32 (4 bytes)
    print(f"\n📊 Model statistics:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Memory (float32): {param_size:.2f} MB")