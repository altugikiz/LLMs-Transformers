import torch
import torch.nn as nn
import torch.optim as optim
from engine.tokenizer import MiniTokenizer
from engine.model import MiniTransformer
import os

# --- 1. VERİ YÜKLEME ---
en_sentences = []
tr_sentences = []

if not os.path.exists("data/corpus.txt"):
    print("Hata: data/corpus.txt bulunamadı!")
    exit()

with open("data/corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|" in line:
            parts = line.strip().split(" | ")
            if len(parts) == 2:
                en, tr = parts
                en_sentences.append(en)
                tr_sentences.append(tr)

# --- 2. TOKENIZER KURULUMU ---
en_tokenizer = MiniTokenizer()
en_tokenizer.build_vocab(en_sentences)

tr_tokenizer = MiniTokenizer()
tr_tokenizer.build_vocab(tr_sentences)

# --- 3. MODEL VE EĞİTİM AYARLARI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışan Cihaz: {device}")

d_model = 64
nhead = 4
num_layers = 2
learning_rate = 0.001
epochs = 100

model = MiniTransformer(
    src_vocab_size=len(en_tokenizer.vocab), 
    trg_vocab_size=len(tr_tokenizer.vocab), 
    d_model=d_model, 
    nhead=nhead, 
    num_layers=num_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tr_tokenizer.vocab["[PAD]"])

# --- 4. GERÇEK EĞİTİM DÖNGÜSÜ ---
print("\nEğitim Başlıyor...")
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for en_sent, tr_sent in zip(en_sentences, tr_sentences):
        src = torch.tensor([en_tokenizer.encode(en_sent)]).to(device)
        trg_full = torch.tensor([tr_tokenizer.encode(tr_sent)]).to(device)
        
        trg_input = trg_full[:, :-1]
        trg_expected = trg_full[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, trg_input)
        
        loss = criterion(output.view(-1, output.size(-1)), trg_expected.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(en_sentences):.4f}")

# --- 5. GERÇEK ÇEVİRİ (INFERENCE) ---
print("\n--- TEST VE ÇEVİRİ ---")
model.eval()

def translate(sentence):
    src = torch.tensor([en_tokenizer.encode(sentence)]).to(device)
    # [BOS] token'ı ile başla
    trg_ids = [tr_tokenizer.vocab["[BOS]"]]
    
    for _ in range(20):
        # trg_ids listesini her adımda tensor'a çeviriyoruz
        trg_input = torch.tensor([trg_ids]).to(device)
        with torch.no_grad():
            output = model(src, trg_input)
            next_token = output.argmax(dim=-1)[:, -1].item()
            trg_ids.append(next_token)
            
            if next_token == tr_tokenizer.vocab["[EOS]"]:
                break
                
    return tr_tokenizer.decode(trg_ids)

# Final Testi
test_sentence = "A cute teddy bear is reading."
result = translate(test_sentence)

print(f"Giriş (EN): {test_sentence}")
print(f"Çıkış (TR): {result}")