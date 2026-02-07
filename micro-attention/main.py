import numpy as np
import matplotlib.pyplot as plt
from attention_engine import ScaledDotProductAttention

# Basit bir "sözlük" ve embedding (temsili değerler)
words = ["Altuğ", "kütüphanede", "ders", "çalışıyor"]
d_model = 4  # Her kelimeyi 4 boyutlu bir vektör olarak düşünelim

# Rastgele ama sabit embeddingler (Gerçekte bunlar eğitilir)
np.random.seed(42)
embeddings = np.random.rand(len(words), d_model)

# Q, K, V matrislerini oluştur (Basitlik için embeddingleri kullanıyoruz)
Q = K = V = embeddings

# Attention hesapla
attention = ScaledDotProductAttention(d_model)
output, weights = attention.forward(Q, K, V)

# GÖRSELLEŞTİRME
plt.figure(figsize=(8, 6))
plt.imshow(weights, cmap='viridis')
plt.xticks(range(len(words)), words, rotation=45)
plt.yticks(range(len(words)), words)
plt.colorbar(label="Attention Score")
plt.title("Kelime İlişki Haritası (Self-Attention Map)")
plt.show()