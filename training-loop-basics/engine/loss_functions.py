import numpy as np

def cross_entropy_loss(probs, target_idx):
    """
    probs: Modelin softmax çıktısı (olasılık dağılımı)
    target_idx: Gerçekte olması gereken kelimenin indeksi
    """
    # Kayıp formülü: -log(doğru sınıfın olasılığı)
    # Model doğru kelimeye ne kadar düşük olasılık verdiyse, ceza o kadar büyük olur.
    loss = -np.log(probs[target_idx] + 1e-9) # 1e-9 sayısal kararlılık için (log 0 olmasın diye)
    return loss