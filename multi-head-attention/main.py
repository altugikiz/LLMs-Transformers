import numpy as np
import matplotlib.pyplot as plt
from engine.mha_layer import MultiHeadAttention

# Parametreler
words = ["Altuğ", "loves", "learning", "about", "AI"]
d_model = 16
num_heads = 4
np.random.seed(7)

# Rastgele Girdi Oluştur (Batch_size=1, Seq_len=5, d_model=16)
input_embeddings = np.random.rand(1, len(words), d_model)

# MHA Uygula
mha = MultiHeadAttention(d_model, num_heads)
output, weights = mha.forward(input_embeddings, input_embeddings, input_embeddings)

# 4 Kafanın Dikkat Haritasını Çiz
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(num_heads):
    ax = axes[i]
    im = ax.imshow(weights[0, i], cmap='magma')
    ax.set_title(f"Head {i+1}")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)

plt.tight_layout()
plt.show()