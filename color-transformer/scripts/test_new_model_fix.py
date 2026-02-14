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
    d_model=128,
    n_heads=8,
    n_layers=4,
    d_ff=512,
    max_seq_len=20
)

checkpoint = torch.load("experiments/exp_20260213_211249/checkpoints/best.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("🎨 YENİ MODEL - FARKLI PARAMETRELER")
print("="*60)

test_prompts = [
    "red apple",
    "blue sky", 
    "green forest"
]

# Farklı parametre kombinasyonları
param_combinations = [
    (0.8, 40, "Normal"),      # Standart
    (1.2, 60, "Yaratıcı"),     # Yüksek temperature
    (1.5, 80, "Çok Yaratıcı"), # Çok yüksek temperature
    (0.5, 20, "Kesin"),        # Düşük temperature
    (1.0, 97, "Max Çeşitlilik") # Tüm vocabulary
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
    print("-"*40)
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    for temp, k, desc in param_combinations:
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_length=10,
                temperature=temp,
                top_k=k,
                eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
            )
        
        result = tokenizer.decode(generated[0].tolist())
        print(f"  {desc:15} (t={temp}): {result}")