import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MiniTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding & Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. PyTorch'un hazır Transformer yapısı (Encoder + Decoder)
        # Slayt 31-35'teki tüm yapıyı kapsar
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True
        )
        
        # 3. Output Head
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg):
        # Kaynak ve Hedef maskeleme (Slayt 24 - Masking)
        src_emb = self.pos_encoder(self.src_embedding(src))
        trg_emb = self.pos_encoder(self.trg_embedding(trg))
        
        # Decoder'ın geleceği görmemesi için maske
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(src.device)
        
        out = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
        return self.fc_out(out)