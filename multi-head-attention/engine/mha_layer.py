import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model head sayısına tam bölünmeli!"

    def split_heads(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, q, k, v):
        # Q, K, V'yi kafalara böl
        qs = self.split_heads(q)
        ks = self.split_heads(k)
        vs = self.split_heads(v)

        # Scaled Dot-Product Attention
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scaled_dot_product = np.matmul(qs, ks.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = softmax(scaled_dot_product)
        
        # Output: (batch_size, num_heads, seq_len, d_k)
        output = np.matmul(attention_weights, vs)
        
        # Kafaları birleştir (Concatenate)
        output = output.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], self.d_model)
        
        return output, attention_weights