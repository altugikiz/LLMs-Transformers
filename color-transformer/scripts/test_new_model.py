# scripts/test_new_model.py
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

# Yeni modeli yükle
tokenizer = ColorTokenizer()
tokenizer.load(Path("data/processed/tokenizer.json"))

model = ColorTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,      # Yeni model boyutları
    n_heads=8,
    n_layers=4,
    d_ff=512,
    max_seq_len=20
)

checkpoint = torch.load("experiments/exp_20260213_211249/checkpoints/best.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("🎨 YENİ MODEL (50 epoch) - %57.7 Accuracy")
print("="*60)

test_prompts = [
    "red apple",
    "blue sky",
    "green forest",
    "yellow sun",
    "purple flower",
    "dark night",
    "white snow",
    "black cat"
]

for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_length=8,
            temperature=0.7,  # Biraz daha düşük temperature artık
            top_k=30,
            eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
        )
    
    result = tokenizer.decode(generated[0].tolist())
    print(f"\n📝 {prompt:15} -> {result}")