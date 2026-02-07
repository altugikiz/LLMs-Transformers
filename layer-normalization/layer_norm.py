import numpy as np

class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        # Öğrenilebilir parametreler (Gamma ölçekler, Beta kaydırır)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        
        # 1. Ortalama hesapla (Son boyut boyunca: d_model)
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # 2. Varyans hesapla
        variance = np.var(x, axis=-1, keepdims=True)
        
        # 3. Normalizasyon (Standardize)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        # 4. Ölçeklendirme ve Kaydırma
        output = self.gamma * x_norm + self.beta
        
        return output