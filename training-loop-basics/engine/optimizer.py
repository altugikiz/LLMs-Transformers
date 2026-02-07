class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, weights, gradients):
        """
        Ağırlıkları gradyan yönünün tersine, öğrenme hızı kadar günceller.
        W = W - (learning_rate * gradient)
        """
        return weights - (self.lr * gradients)