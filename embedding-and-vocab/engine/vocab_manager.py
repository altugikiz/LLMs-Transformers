class VocabManager:
    def __init__(self):
        # Özel tokenlar: Padding, Unknown, Start of Sentence, End of Sentence
        self.stoi = { "<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3 }
        self.itos = { 0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>" }
        self.vocab_size = 4

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = self.vocab_size
                self.itos[self.vocab_size] = token
                self.vocab_size += 1

    def encode(self, tokens):
        # Token listesini ID listesine çevirir
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in tokens]

    def decode(self, ids):
        # ID listesini tekrar kelimelere çevirir
        return [self.itos.get(i, "<UNK>") for i in ids]