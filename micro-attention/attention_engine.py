import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class ScaledDotProductAttention:
    def __init__(self, d_model):
        self.d_k = d_model
        
    def forward(self, queries, keys, values):
        # 1. Matris Çarpımı (Q * K^T)
        # Skorlar kelimelerin birbirine ne kadar "benzediğini" ölçer
        scores = np.matmul(queries, keys.T) / np.sqrt(self.d_k)
        
        # 2. Softmax ile olasılık dağılımı oluşturma
        attention_weights = softmax(scores)
        
        # 3. Değerler (Values) ile ağırlıklı toplam alma
        output = np.matmul(attention_weights, values)
        
        return output, attention_weights