import numpy as np

def softmax(x):
    # Sayısal kararlılık için max çıkarılır
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class OutputLayer:
    def __init__(self, d_model, vocab_size):
        # Ağırlık matrisi: (d_model, vocab_size)
        self.W = np.random.randn(d_model, vocab_size) * 0.1
        self.b = np.zeros(vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        
        # 1. Linear: Vektörü sözlük boyutuna genişlet
        logits = np.dot(x, self.W) + self.b
        
        # 2. Softmax: Olasılıkları hesapla
        probabilities = softmax(logits)
        
        return probabilities, logits