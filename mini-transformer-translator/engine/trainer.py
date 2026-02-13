import numpy as np

class MiniTrainer:
    def __init__(self, model, vocab_size, learning_rate=0.01):
        self.model = model
        self.vocab_size = vocab_size
        self.lr = learning_rate
        # Çıkış katmanı ağırlıkları (Logits üretmek için)
        self.W_out = np.random.randn(model.d_model, vocab_size) * 0.1

    def cross_entropy_loss(self, probs, target_idx):
        # Slayt 8: Modelin hatasını ölçer
        return -np.log(probs[target_idx] + 1e-9)

    def train_step(self, input_ids, target_ids):
        """
        Basitleştirilmiş tek bir eğitim adımı.
        """
        # 1. Forward Pass (Transformer katmanlarından geçir)
        x = self.model.get_embeddings(input_ids)
        # Basitlik adına tek bir bloktan geçiriyoruz
        hidden_states, _ = self.model.forward(x) 
        
        # 2. Output Head: Vektörleri sözlük boyutuna çıkar
        logits = np.dot(hidden_states, self.W_out)
        # Softmax ile olasılıklara dönüştür
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        # 3. Hata (Loss) Hesapla
        total_loss = 0
        for i, target_id in enumerate(target_ids):
            if i < len(probs):
                total_loss += self.cross_entropy_loss(probs[i], target_id)
        
        # 4. Basit Gradyan İnişi (Ağırlıkları güncelleme simülasyonu)
        # Not: Gerçekte burada Backpropagation (Zincir Kuralı) çalışır.
        self.W_out -= self.lr * 0.01 # Temsili güncelleme
        
        return total_loss / len(target_ids)