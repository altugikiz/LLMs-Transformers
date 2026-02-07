import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Başlangıçta rastgele ağırlıklar (Eğitimle optimize edilir)
        # d_model: Her kelimenin kaç boyutlu vektörle temsil edileceği
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, token_ids):
        # ID'leri kullanarak tablodan ilgili vektörleri seçer (Lookup Table)
        return self.embeddings[token_ids]