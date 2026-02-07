import numpy as np
import matplotlib.pyplot as plt
from pos_encoding import PositionalEncoding

# Parametreler
SEQ_LEN = 100  # Cümle uzunluğu
D_MODEL = 128  # Vektör boyutu

# Encoding oluştur
pe_gen = PositionalEncoding(SEQ_LEN, D_MODEL)
pe_matrix = pe_gen.get_encoding()

# Görselleştir
plt.figure(figsize=(12, 8))
plt.pcolormesh(pe_matrix, cmap='RdBu')
plt.xlabel('Embedding Dimension (d_model)')
plt.ylabel('Sequence Position')
plt.colorbar(label='Encoding Value')
plt.title("Positional Encoding Matrix (Sin/Cos Patterns)")
plt.show()