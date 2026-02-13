from engine.tokenizer import MiniTokenizer

# 1. Veriyi Oku
en_sentences = []
tr_sentences = []

with open("data/corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        en, tr = line.strip().split(" | ")
        en_sentences.append(en)
        tr_sentences.append(tr)

# 2. Tokenizer'ları Başlat
# Gerçek modellerde genelde tek bir tokenizer her şeyi halleder ama 
# biz dilleri ayırmak için iki tane kurabiliriz.
en_tokenizer = MiniTokenizer()
en_tokenizer.build_vocab(en_sentences)

tr_tokenizer = MiniTokenizer()
tr_tokenizer.build_vocab(tr_sentences)

# 3. Test Et: "A cute teddy bear is reading."
test_sentence = en_sentences[0]
encoded = en_tokenizer.encode(test_sentence)
decoded = en_tokenizer.decode(encoded)

print(f"\nOrijinal: {test_sentence}")
print(f"Token IDs: {encoded}")
print(f"Decoding: {decoded}")



import matplotlib.pyplot as plt
from engine.transformer import MiniTransformerComponents, MultiHeadAttention

# Tokenizer kısmından gelen encoded cümleyi kullanıyoruz
# Örn: [BOS, a, cute, teddy, bear, is, read, ##ing, [EOS]]
d_model = 16
vocab_size = len(en_tokenizer.vocab)

# 1. Parçaları Başlat
components = MiniTransformerComponents(vocab_size, d_model)
attention = MultiHeadAttention(d_model, num_heads=2)

# 2. Vektörleri Al (Embedding + Position)
x = components.get_embeddings(encoded)

# 3. Attention Mekanizmasından Geçir
output, weights = attention.forward(x)

# 4. GÖRSELLEŞTİRME (Slayt 20'deki Isı Haritası)
tokens = [en_tokenizer.itos[i] for i in encoded]
plt.figure(figsize=(10, 8))
plt.imshow(weights, cmap='viridis')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.title("Attention Map: 'A cute teddy bear is reading'")
plt.colorbar()
plt.show()