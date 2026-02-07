import numpy as np
from engine.loss_functions import cross_entropy_loss
from engine.optimizer import SGDOptimizer

# 1. Senaryo: Gerçek kelime 'banana' (indeks 4) olsun
target_idx = 4
learning_rate = 0.1

# 2. Modelin hatalı tahmini (Output Head'den gelen olasılıklar)
# Model banana'ya %10, apple'a %50 vermiş olsun (Hatalı durum)
probs = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.50, 0.05, 0.05])
W_output = np.random.randn(16, 9) # Temsili ağırlık matrisi

print(f"Başlangıç Kaybı: {cross_entropy_loss(probs, target_idx):.4f}")

# 3. Backpropagation (Basitleştirilmiş Gradyan Hesabı)
# Matematiksel olarak Softmax + CrossEntropy gradyanı: (Tahmin - Gerçek)
target_one_hot = np.zeros(9)
target_one_hot[target_idx] = 1.0
error_gradient = probs - target_one_hot # Hata vektörü

# 4. Güncelleme
optimizer = SGDOptimizer(learning_rate==learning_rate)
# Ağırlıkların nasıl güncellendiğini simüle edelim (Basit bir gradyan adımı)
# Gerçekte bu işlem her bir parametre için türev zinciri (Chain Rule) ile yapılır.
W_output_updated = optimizer.step(W_output, np.outer(np.random.randn(16), error_gradient))

print("Ağırlıklar güncellendi. Bir sonraki tahminde 'banana' olasılığı artacak!")