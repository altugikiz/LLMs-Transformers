from bpe_tokenizer import SimpleBPETokenizer

tokenizer = SimpleBPETokenizer()
text = "banana banana bandana"

print(f"Orijinal Metin: {text}")
tokens = tokenizer.tokenize_demo(text, num_merges=5)

print(f"\nFinal Token Sayısı: {len(tokens)}")
# Normalde 21 karakter varken, birleştirmelerle token sayısı düşer.