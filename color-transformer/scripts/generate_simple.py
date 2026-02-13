# scripts/generate_simple.py
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

# Model ve tokenizer'ı yükle
tokenizer = ColorTokenizer()
tokenizer.load(Path("data/processed/tokenizer.json"))

model = ColorTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_seq_len=20
)
checkpoint = torch.load("experiments/exp_20260213_204243/checkpoints/best.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test edilecek prompt'lar
test_prompts = [
    "red apple",
    "blue sky", 
    "green forest",
    "yellow sun",
    "purple flower",
    "white snow"
]

print("\n🎨 RENK TRANSFORMER - ÜRETİM ÖRNEKLERİ")
print("="*60)

for prompt in test_prompts:
    # Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    # Farklı temperature değerleriyle üret
    with torch.no_grad():
        # Kesin (temperature düşük)
        gen1 = model.generate(
            input_tensor,
            max_length=8,
            temperature=0.3,
            top_k=10,
            eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
        )
        
        # Normal (temperature orta)
        gen2 = model.generate(
            input_tensor,
            max_length=8,
            temperature=0.8,
            top_k=40,
            eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
        )
        
        # Yaratıcı (temperature yüksek)
        gen3 = model.generate(
            input_tensor,
            max_length=8,
            temperature=1.2,
            top_k=50,  # 97'den küçük!
            eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
        )
    
    print(f"\n📝 {prompt}:")
    print(f"  Kesin    : {tokenizer.decode(gen1[0].tolist())}")
    print(f"  Normal   : {tokenizer.decode(gen2[0].tolist())}")
    print(f"  Yaratıcı : {tokenizer.decode(gen3[0].tolist())}")