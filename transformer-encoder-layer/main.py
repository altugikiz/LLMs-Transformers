import numpy as np
from transformer_block import TransformerEncoderLayer
from engine.positional_encoding import PositionalEncoding

# Konfigürasyon
words = ["Altuğ", "is", "building", "a", "transformer"]
d_model = 32
num_heads = 4
d_ff = 128
seq_len = len(words)

# 1. Giriş: Rastgele Embeddingler (Normalde eğitilmiş vektörler olur)
input_embeddings = np.random.randn(1, seq_len, d_model)

# 2. Positional Encoding Ekleme
pe = PositionalEncoding(seq_len, d_model).get_encoding()
x = input_embeddings + pe

# 3. Transformer Layer'dan Geçirme
encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
output = encoder_layer.forward(x)

print(f"Final Output Shape: {output.shape}") # (1, 5, 32)
print("İşlem başarıyla tamamlandı. Kelimeler artık derin bağlamsal vektörlere dönüştü.")

import matplotlib.pyplot as plt

# Encoder çıkışını görselleştir
plt.figure(figsize=(10, 4))
plt.imshow(output[0], aspect='auto', cmap='inferno')
plt.title("Transformer Encoder Output (Contextual Embeddings)")
plt.xlabel("Dimension")
plt.ylabel("Token Index")
plt.colorbar()
plt.show()