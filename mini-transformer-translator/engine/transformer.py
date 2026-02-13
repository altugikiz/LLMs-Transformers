import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class MiniTransformerComponents:
    def __init__(self, vocab_size, d_model, max_len=100):
        self.d_model = d_model
        
        # 1. Embedding Katmanı (Slayt 12)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        
        # 2. Positional Encoding (Slayt 14, 22)
        self.pos_encoding = self._generate_positional_encoding(max_len, d_model)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div_term = np.exp(i * -(np.log(10000.0) / d_model))
                pe[pos, i] = np.sin(pos * div_term)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos * div_term)
        return pe

    def get_embeddings(self, token_ids):
        # Kelime vektörleri + Pozisyon vektörleri (Slayt 22)
        seq_len = len(token_ids)
        x = self.embedding[token_ids]
        x += self.pos_encoding[:seq_len, :]
        return x

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V için rastgele ağırlıklar (Slayt 17-20)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        # x: (seq_len, d_model)
        # 1. Q, K, V Matrislerini oluştur
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. Scaled Dot-Product Attention (Slayt 18)
        # Scores = (Q * K^T) / sqrt(d_k)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        attn_weights = softmax(scores)
        
        # 3. Sonuç: Weights * V
        output = np.dot(attn_weights, V)
        return output, attn_weights