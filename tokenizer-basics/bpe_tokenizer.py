import collections

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = {}
        
    def get_stats(self, ids):
        # Yan yana gelen çiftlerin frekansını sayar
        counts = collections.defaultdict(int)
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def merge(self, ids, pair, idx):
        # Belirlenen çifti yeni bir ID ile birleştirir
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def tokenize_demo(self, text, num_merges=3):
        # Metni başlangıçta karakter ID'lerine (ASCII) ayır
        ids = list(text.encode("utf-8"))
        print(f"Başlangıç ID'leri: {ids}")
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            # En çok tekrar eden çifti bul
            best = max(stats, key=stats.get)
            new_id = 256 + i # ASCII sonrası ilk boş ID
            ids = self.merge(ids, best, new_id)
            print(f"Merge {i+1}: {best} -> {new_id} | Yeni ID'ler: {ids}")
        
        return ids