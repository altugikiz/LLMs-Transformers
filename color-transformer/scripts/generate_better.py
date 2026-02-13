# scripts/generate_better.py
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

def generate_with_params(prompt, temperature, top_k):
    """Farklı parametrelerle üretim yap"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    # top_k vocabulary boyutundan büyük olamaz!
    vocab_size = tokenizer.vocab_size
    if top_k > vocab_size:
        print(f"⚠️ top_k={top_k} > vocab_size={vocab_size}, düzeltiliyor...")
        top_k = vocab_size
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_length=8,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
        )
    
    return tokenizer.decode(generated[0].tolist())

# Model ve tokenizer'ı yükle
print("✅ Tokenizer yükleniyor...")
tokenizer = ColorTokenizer()
tokenizer.load(Path("data/processed/tokenizer.json"))
print(f"   Vocabulary size: {tokenizer.vocab_size}")

print("\n✅ Model yükleniyor...")
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
print(f"   Model epoch: {checkpoint['epoch']}")

# Test prompts
prompts = ["red apple", "blue sky", "green forest"]

print("\n🔬 PARAMETRE TESTİ")
print("="*60)

for prompt in prompts:
    print(f"\n📝 Prompt: {prompt}")
    print("-"*40)
    
    # Farklı parametre kombinasyonları
    params_list = [
        (0.3, 10, "Kesin (düşük temperature, az çeşitlilik)"),
        (0.8, 40, "Normal (orta temperature, orta çeşitlilik)"),
        (1.2, 50, "Yaratıcı (yüksek temperature, çok çeşitlilik)"),
        (0.5, 5,  "Çok Kesin (çok düşük çeşitlilik)"),
        (1.5, 70, "Çok Yaratıcı (çok yüksek temperature)"),
    ]
    
    for temp, k, desc in params_list:
        try:
            result = generate_with_params(prompt, temp, k)
            print(f"  {desc}: {result}")
        except Exception as e:
            print(f"  {desc}: HATA - {e}")

print("\n✅ Test tamamlandı!")