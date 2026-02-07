import numpy as np
import matplotlib.pyplot as plt
from layer_norm import LayerNormalization

# 1. Çok dengesiz değerlere sahip bir girdi oluşturalım
# Bazı değerler çok büyük, bazıları çok küçük
np.random.seed(42)
unnormalized_data = np.random.randn(1, 5, 10) * 50 + 100 

# 2. LayerNorm uygula
ln = LayerNormalization(d_model=10)
normalized_data = ln.forward(unnormalized_data)

# 3. Sonuçları Karşılaştır
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(unnormalized_data[0], cmap='coolwarm')
ax1.set_title("Orijinal Veri (Dengesiz)")
ax1.set_xlabel("Feature Dimension")
ax1.set_ylabel("Sequence")

ax2.imshow(normalized_data[0], cmap='coolwarm')
ax2.set_title("Normalize Edilmiş Veri (Dengeli)")
ax2.set_xlabel("Feature Dimension")

plt.tight_layout()
plt.show()

print(f"Orijinal Ortalama: {np.mean(unnormalized_data):.2f}")
print(f"Normalize Ortalama: {np.mean(normalized_data):.2f} (Yaklaşık 0)")