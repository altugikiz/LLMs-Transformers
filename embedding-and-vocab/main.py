import numpy as np
from engine.vocab_manager import VocabManager
from engine.embedding_layer import EmbeddingLayer

# 1. Hazırlık
vm = VocabManager()
corpus = ["banana", "bandana", "apple", "intelligence", "llm"]
vm.add_tokens(corpus)

# 2. Örnek bir cümle (Tokenize edilmiş varsayalım)
input_tokens = ["banana", "llm", "unknown_word"]
token_ids = vm.encode(input_tokens)

print(f"Sözlük Boyutu: {vm.vocab_size}")
print(f"Token ID'leri: {token_ids}  (<UNK> id: 1)")

# 3. Embedding Süreci
d_model = 16 # Her kelimeyi 16 boyutlu temsil et
emb = EmbeddingLayer(vm.vocab_size, d_model)
vectors = emb.forward(token_ids)

print(f"\nEmbedding Çıktı Boyutu: {vectors.shape}") # (3, 16)
print(f"'banana' kelimesinin vektörü:\n{vectors[0]}")

# 4. Geriye Dönüş (Decoding)
decoded = vm.decode(token_ids)
print(f"\nID'den Geriye Dönüş: {decoded}")