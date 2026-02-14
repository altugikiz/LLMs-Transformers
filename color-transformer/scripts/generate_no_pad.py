# scripts/generate_no_pad.py
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

def generate_no_pad(model, tokenizer, prompt, temperature=1.0):
    """PAD token'ını engelleyerek üret"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    pad_token_id = tokenizer.word2idx[tokenizer.pad_token]  # <PAD> token ID'si
    
    print(f"\n🎯 Prompt: '{prompt}' (PAD engellendi)")
    print("="*50)
    
    with torch.no_grad():
        for step in range(8):
            logits = model.forward(input_tensor)
            next_token_logits = logits[:, -1, :].clone()
            
            # PAD token'ını engelle
            next_token_logits[:, pad_token_id] = float('-inf')
            
            # Temperature uygula
            next_token_logits = next_token_logits / temperature
            
            # Prob'lara çevir
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top 5 göster
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            print(f"\nStep {step+1}:")
            print(f"  Top 5 predictions:")
            for i in range(5):
                token = tokenizer.decode([top_indices[i].item()])
                print(f"    {token:15} : {top_probs[i].item():.4f}")
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Şu ana kadar üretilen
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

# Dene
generate_no_pad(model, tokenizer, "red apple", temperature=1.2)