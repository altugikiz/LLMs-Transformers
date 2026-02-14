# scripts/generate_final_fix.py
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

def generate_with_blocked_tokens(model, tokenizer, prompt, temperature=1.2):
    """Tüm özel token'ları engelleyerek üret"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])  # Shape: (1, seq_len)
    
    # Engellenecek token'lar
    pad_token_id = tokenizer.word2idx[tokenizer.pad_token]
    bos_token_id = tokenizer.word2idx[tokenizer.bos_token]
    eos_token_id = tokenizer.word2idx[tokenizer.eos_token]
    unk_token_id = tokenizer.word2idx[tokenizer.unk_token]
    
    blocked_tokens = [pad_token_id, bos_token_id, eos_token_id, unk_token_id]
    
    print(f"\n🎯 Prompt: '{prompt}'")
    print(f"🚫 Blocked tokens: {[tokenizer.decode([t]) for t in blocked_tokens]}")
    print("="*60)
    
    with torch.no_grad():
        for step in range(10):  # 10 token üret
            logits = model.forward(input_tensor)
            next_token_logits = logits[:, -1, :].clone()  # Shape: (1, vocab_size)
            
            # Tüm özel token'ları engelle
            for token_id in blocked_tokens:
                next_token_logits[:, token_id] = float('-inf')
            
            # Temperature uygula
            next_token_logits = next_token_logits / temperature
            
            # Softmax
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top 5 göster
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            print(f"\nStep {step+1}:")
            print(f"  Top 5 predictions:")
            for i in range(5):
                token_id = top_indices[i].item()
                token = tokenizer.decode([token_id])
                prob = top_probs[i].item()
                print(f"    {token:15} : {prob:.4f}")
            
            # Sample - top 30'dan seç (daha fazla çeşitlilik)
            top_k = min(30, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # Normalize et
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Seçim yap
            chosen_idx = torch.multinomial(top_probs[0], 1)
            next_token_id = top_indices[0, chosen_idx]  # Shape: (1,)
            
            # Doğru boyuta getir: (1, 1)
            next_token = next_token_id.unsqueeze(0)  # Shape: (1, 1)
            
            # Birleştir
            input_tensor = torch.cat([input_tensor, next_token], dim=1)  # Shape: (1, seq_len+1)
            
            current = tokenizer.decode(input_tensor[0].tolist())
            print(f"  Current: {current}")
            
            # Eğer aynı token'ı 3 kere üretirse dur
            if step > 2:
                last_tokens = input_tensor[0, -3:].tolist()
                if len(set(last_tokens)) == 1:
                    print(f"  🛑 Aynı token 3 kere üretildi, durduruluyor")
                    break

# Test
print("✅ Tokenizer yükleniyor...")
tokenizer = ColorTokenizer()
tokenizer.load(Path("data/processed/tokenizer.json"))

print("\n✅ Model yükleniyor...")
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
for temp in [1.2, 1.5, 2.0]:
    print(f"\n{'#'*60}")
    print(f"# TEMPERATURE = {temp}")
    print(f"{'#'*60}")
    generate_with_blocked_tokens(model, tokenizer, "red apple", temperature=temp)
    
print("\n✅ Test tamamlandı!")