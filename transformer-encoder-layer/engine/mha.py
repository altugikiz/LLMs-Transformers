import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, q, k, v):
        qs = self.split_heads(q)
        ks = self.split_heads(k)
        vs = self.split_heads(v)

        # Scaled Dot-Product Attention
        # (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) -> (batch, heads, seq, seq)
        scores = np.matmul(qs, ks.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        weights = softmax(scores)
        
        # (batch, heads, seq, seq) @ (batch, heads, seq, d_k) -> (batch, heads, seq, d_k)
        context = np.matmul(weights, vs)
        
        # Concatenate heads
        output = context.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], self.d_model)
        return output, weights