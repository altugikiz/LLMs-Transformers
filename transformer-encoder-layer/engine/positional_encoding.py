import numpy as np

class PositionalEncoding:
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model
        
    def get_encoding(self):
        pe = np.zeros((self.seq_len, self.d_model))
        for pos in range(self.seq_len):
            for i in range(0, self.d_model, 2):
                div_term = np.exp(i * -(np.log(10000.0) / self.d_model))
                pe[pos, i] = np.sin(pos * div_term)
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = np.cos(pos * div_term)
        return pe