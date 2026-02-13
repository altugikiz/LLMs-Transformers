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