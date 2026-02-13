# scripts/generate.py
import torch
import sys
from pathlib import Path

# Proje kök dizinini ekle
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ColorTransformer
from src.data.tokenizer import ColorTokenizer

def main():
    print("🧪 Generating text with trained model...")
    
    # Yollar
    checkpoint_dir = Path("experiments/exp_20260213_204243/checkpoints")
    tokenizer_path = Path("data/processed/tokenizer.json")
    
    # Tokenizer'ı yükle
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("   Please save tokenizer first by running:")
        print("   from src.data.dataset import ColorSentenceDataset")
        print("   dataset = ColorSentenceDataset(data_path='data/raw/color_sentences_5000.csv', split='train')")
        print("   dataset.get_tokenizer().save('data/processed/tokenizer.json')")
        return
    
    tokenizer = ColorTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"✅ Tokenizer loaded: {tokenizer.vocab_size} tokens")
    
    # En iyi checkpoint'i bul
    best_checkpoint = checkpoint_dir / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = checkpoint_dir / "latest.pt"
    
    if not best_checkpoint.exists():
        print(f"❌ No checkpoint found in {checkpoint_dir}")
        return
    
    print(f"✅ Loading checkpoint: {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    
    # Model oluştur (eğitimdeki config ile aynı olmalı)
    model = ColorTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,      # training_config.yaml'deki değerler
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=20,
        dropout=0.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    
    # Test prompts
    test_prompts = [
        "red apple",
        "blue sky",
        "green forest",
        "yellow sun",
        "dark night"
    ]
    
    print("\n" + "="*50)
    print("GENERATIONS:")
    print("="*50)
    
    for prompt in test_prompts:
        # Tokenize
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids])
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_length=10,
                temperature=0.8,
                top_k=40,
                eos_token_id=tokenizer.word2idx[tokenizer.eos_token]
            )
        
        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"\n📝 Prompt: {prompt}")
        print(f"   Generated: {generated_text}")

if __name__ == "__main__":
    main()