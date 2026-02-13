# scripts/generate.py
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

# Model ve tokenizer'ı yükle
checkpoint = torch.load('experiments/exp_20260213_204243/checkpoints/best.pt')
tokenizer = ColorTokenizer()
tokenizer.load('data/processed/tokenizer.json')  # Tokenizer'ı kaydetmeyi unutma!

model = ColorTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_seq_len=20
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test et
prompt = "red apple"  # Renk + nesne
input_ids = tokenizer.encode(prompt, add_special_tokens=True)
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    generated = model.generate(
        input_tensor,
        max_length=10,
        temperature=0.8,
        top_k=40
    )
    
print(f"Generated: {tokenizer.decode(generated[0].tolist())}")