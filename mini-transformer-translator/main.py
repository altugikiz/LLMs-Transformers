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
            en, tr = line.strip().split(" | ")
            en_sentences.append(en)
            tr_sentences.append(tr)

# --- 2. TOKENIZER KURULUMU ---
en_tokenizer = MiniTokenizer()
en_tokenizer.build_vocab(en_sentences)

tr_tokenizer = MiniTokenizer()
tr_tokenizer.build_vocab(tr_sentences)

# --- 3. MODEL VE EĞİTİM AYARLARI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

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
criterion = nn.CrossEntropyLoss(ignore_index=en_tokenizer.vocab["[PAD]"])

# --- 4. GERÇEK EĞİTİM DÖNGÜSÜ ---
print("\nEğitim Başlıyor...")
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for en_sent, tr_sent in zip(en_sentences, tr_sentences):
        # Sayısal dizilere çevir (Tensor)
        src = torch.tensor([en_tokenizer.encode(en_sent)]).to(device)
        trg_full = torch.tensor([tr_tokenizer.encode(tr_sent)]).to(device)
        
        # Teacher Forcing: Giriş ve Beklenen Çıktı ayrımı
        # trg_input: [BOS, Sevimli, bir, ...]
        # trg_expected: [Sevimli, bir, ..., EOS]
        trg_input = trg_full[:, :-1]
        trg_expected = trg_full[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward Pass
        output = model(src, trg_input)
        
        # Loss hesaplama (Boyutları düzeltiyoruz: [Batch*Seq, Vocab])
        loss = criterion(output.view(-1, output.size(-1)), trg_expected.view(-1))
        
        # Backpropagation (Sihirli an!)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(en_sentences):.4f}")

# --- 5. GERÇEK ÇIKARIM (INFERENCE) ---
print("\n--- TEST AŞAMASI ---")
model.eval()

def translate(sentence):
    src = torch.tensor([en_tokenizer.encode(sentence)]).to(device)
    # Başlangıç token'ı ile başla
    trg_input = torch.tensor([[tr_tokenizer.vocab["[BOS]"]]]).to(device)
    
    for _ in range(20): # Maksimum 20 kelime üret
        with torch.no_grad():
            output = model(src, trg_input)
            # En son üretilen kelimenin ID'sini al (Greedy Search)
            next_token = output.argmax(dim=-1)[:, -1].item()
            
            # Yeni kelimeyi girişe ekle
            trg_input = torch.cat([trg_input, torch.tensor([[next_token]]).to(device)], dim=1)
            
            # Eğer [EOS] üretildiyse dur
            if next_token == trg_tokenizer.vocab["[EOS]"]:
                break
                
    return tr_tokenizer.decode(trg_input[0].tolist())

# Final Testi
test_sentence = "A cute teddy bear is reading."
result = translate(test_sentence)

print(f"İngilizce: {test_sentence}")
print(f"Modelin Çevirisi: {result}")