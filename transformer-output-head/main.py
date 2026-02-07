import numpy as np
from engine.output_layer import OutputLayer

# 1. Hazırlık (Daha önceki aşamalardan gelen veriler gibi düşün)
vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "banana", "bandana", "apple", "intelligence", "llm"]
d_model = 16
vocab_size = len(vocab)

# Transformer'dan çıkmış temsili bir vektör (1 kelime için)
# (batch_size=1, seq_len=1, d_model=16)
transformer_output = np.random.randn(1, 1, d_model)

# 2. Output Head İşlemi
output_head = OutputLayer(d_model, vocab_size)
probs, logits = output_head.forward(transformer_output)

# 3. Sonuçları Yorumlama
# En yüksek olasılıklı indeks
predicted_idx = np.argmax(probs[0, 0])
predicted_word = vocab[predicted_idx]

print(f"Sözlükteki Olasılık Dağılımı:\n{probs[0, 0].round(3)}")
print(f"\nEn yüksek olasılıklı indeks: {predicted_idx}")
print(f"Modelin tahmini: '{predicted_word}' (Olasılık: {probs[0, 0][predicted_idx]:.2%})")