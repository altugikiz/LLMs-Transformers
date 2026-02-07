import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        # Küçük rastgele değerlerle başlatma
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # Linear 1 + ReLU
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        # Linear 2
        return np.dot(a1, self.W2) + self.b2