# scripts/generate_aggressive.py
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

def generate_with_debug(model, tokenizer, prompt, temperature=1.0):
    """Debug ile üretim"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    print(f"\n🎯 Prompt: '{prompt}'")
    print("="*50)
    
    with torch.no_grad():
        for step in range(8):  # 8 token üret
            logits = model.forward(input_tensor)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Softmax ile olasılıklara çevir
            probs = F.softmax(next_token_logits, dim=-1)
            
            # En yüksek 3 token'ı göster
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            print(f"\nStep {step+1}:")
            print(f"  Top 5 predictions:")
            for i in range(5):
                token = tokenizer.decode([top_indices[i].item()])
                print(f"    {token:15} : {top_probs[i].item():.4f}")
            
            # Sampling (temperature etkili olsun diye)
            if temperature > 1.0:
                # Daha yaratıcı - tüm token'ları düşün
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Daha kesin - top-5'ten seç
                top5_probs = probs[0][top_indices]
                top5_probs = top5_probs / top5_probs.sum()
                next_token = top_indices[torch.multinomial(top5_probs, 1)]
                next_token = next_token.unsqueeze(0)
            
            # Yeni token'ı ekle
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Şu ana kadar üretilen cümle
            current = tokenizer.decode(input_tensor[0].tolist())
            print(f"  Current: {current}")

# Test
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

# Farklı temperature'larla dene
for temp in [0.5, 1.0, 1.5, 2.0]:
    print(f"\n{'#'*60}")
    print(f"# TEMPERATURE = {temp}")
    print(f"{'#'*60}")
    generate_with_debug(model, tokenizer, "red apple", temperature=temp)