import re
from collections import Counter

class MiniTokenizer:
    def __init__(self):
        # Özel tokenlar: Padding, Unknown, Beginning of Sentence, End of Sentence
        self.special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        self.vocab = {}
        self.itos = {} # Index to String
        
    def _subword_split(self, word):
        """
        Basit sub-word mantığı. 
        'reading' -> ['read', '##ing'] gibi ekleri ayırır.
        """
        suffixes = ['ing', 'ly', 's', 'ed', 'er', 'est']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return [word[:-len(suffix)], f"##{suffix}"]
        return [word]

    def build_vocab(self, sentences):
        # Tüm cümleleri işle ve frekans say
        tokens = []
        for sentence in sentences:
            # Noktalama işaretlerini ayır ve küçük harfe çevir
            clean_text = re.sub(r'([.,!?])', r' \1 ', sentence.lower())
            words = clean_text.split()
            
            for word in words:
                tokens.extend(self._subword_split(word))
        
        # Benzersiz tokenları belirle ve özel tokenları en başa ekle
        unique_tokens = self.special_tokens + sorted(list(set(tokens)))
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.itos = {i: token for token, i in self.vocab.items()}
        print(f"Vocab oluşturuldu! Toplam Token Sayısı: {len(self.vocab)}")

    def encode(self, sentence, add_special=True):
        clean_text = re.sub(r'([.,!?])', r' \1 ', sentence.lower())
        words = clean_text.split()
        
        token_ids = []
        if add_special: token_ids.append(self.vocab["[BOS]"])
        
        for word in words:
            subwords = self._subword_split(word)
            for sw in subwords:
                token_ids.append(self.vocab.get(sw, self.vocab["[UNK]"]))
                
        if add_special: token_ids.append(self.vocab["[EOS]"])
        return token_ids

    def decode(self, ids):
        tokens = [self.itos.get(i, "[UNK]") for i in ids]
        # Özel tokenları temizle ve ## eklerini birleştir
        clean_tokens = []
        for t in tokens:
            if t in self.special_tokens: continue
            if t.startswith("##"):
                if clean_tokens: clean_tokens[-1] += t[2:]
            else:
                clean_tokens.append(t)
        return " ".join(clean_tokens)