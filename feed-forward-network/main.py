import numpy as np
import matplotlib.pyplot as plt
from ffn import FeedForwardNetwork

# Parametreler
d_model = 64
d_ff = 256  # Genelde d_model'in 4 katı olur
seq_len = 10

# Örnek girdi
sample_input = np.random.randn(1, seq_len, d_model)

# FFN Uygula
ffn = FeedForwardNetwork(d_model, d_ff)
output = ffn.forward(sample_input)

# Girdi ve Çıktı arasındaki farkı görselleştir (Ağ veriyi nasıl dönüştürmüş?)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_input[0], aspect='auto', cmap='viridis')
plt.title("Input Embedding")

plt.subplot(1, 2, 2)
plt.imshow(output[0], aspect='auto', cmap='viridis')
plt.title("FFN Output")

plt.tight_layout()
plt.show()

print(f"Girdi Boyutu: {sample_input.shape}")
print(f"Çıktı Boyutu: {output.shape}")