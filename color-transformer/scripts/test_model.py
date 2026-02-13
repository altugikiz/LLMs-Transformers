# scripts/test_model.py
import torch
from src.models.transformer import ColorTransformer

# Minimal model
model = ColorTransformer(
    vocab_size=97,
    d_model=64,  # Daha küçük
    n_heads=4,   # Daha az head
    n_layers=2,  # Daha az layer
    d_ff=256,    # Daha küçük FFN
    max_seq_len=20
)

# Test verisi
input_ids = torch.randint(1, 96, (2, 10))
attention_mask = torch.ones(2, 10)

# Forward pass
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    
    if torch.isnan(logits).any():
        print("❌ NaN detected!")
    else:
        print("✅ No NaN - model works!")