import numpy as np

def relu(x):
    return np.maximum(0, x)

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        # Ağırlıkları başlat (He initialization mantığıyla küçük rastgele sayılar)
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        # 1. İlk Lineer Katman + ReLU
        # (batch, seq, d_model) @ (d_model, d_ff) -> (batch, seq, d_ff)
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        
        # 2. İkinci Lineer Katman
        # (batch, seq, d_ff) @ (d_ff, d_model) -> (batch, seq, d_model)
        output = np.dot(a1, self.W2) + self.b2
        
        return output