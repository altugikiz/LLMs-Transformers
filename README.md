# LLMs-Transformers: Transformer Bileşenlerini Sıfırdan Uygulama 🚀

Bu repo, modern büyük dil modellerinin (LLM) temelini oluşturan Transformer mimarisinin tüm ana bileşenlerini **NumPy** ile sıfırdan, adım adım ve modüler olarak inşa etmektedir. Her klasör, Transformer'ın bir parçasını izole şekilde ele alır ve matematiksel sezgiyi öne çıkarır.

## 📦 Proje Klasörleri ve İçerikleri

- **embedding-and-vocab/**: Kelime gömme (embedding) ve temel kelime dağarcığı işlemleri.
- **tokenizer-basics/**: Byte Pair Encoding (BPE) ile tokenizasyonun temelleri.
- **positional-encoding/**: Sine ve Cosine fonksiyonlarıyla pozisyonel kodlama.
- **micro-attention/**: Scaled Dot-Product Attention'ın saf NumPy ile görselleştirilmiş uygulaması.
- **multi-head-attention/**: Çoklu başlı dikkat mekanizmasının matematiksel olarak bölünmüş ve paralel çalışan versiyonu.
- **layer-normalization/**: Transformer'larda istikrar için Layer Normalization'ın sıfırdan inşası.
- **feed-forward-network/**: Her pozisyona bağımsız uygulanan iki katmanlı doğrusal ağ (FFN).
- **training-loop-basics/**: (Boş veya temel eğitim döngüsü örnekleri için ayrılmıştır.)
- **transformer-encoder-layer/**: Tüm bileşenlerin birleşimiyle tam bir Transformer Encoder Bloğu.
- **transformer-output-head/**: Model çıktısını kelime olasılıklarına dönüştüren çıkış başı.

## 🛠️ Teknolojiler

- **Python 3.x**
- **NumPy**: Tüm matris işlemleri ve doğrusal cebir için
- **Matplotlib**: Dikkat haritaları ve pozisyonel kodlama görselleştirmeleri

## 🚦 Hızlı Başlangıç

Her alt klasörde:
```bash
cd klasor-adi
python main.py
```
Çoğu modül, çalıştırıldığında örnek bir girişle sonucu veya görselleştirmeyi ekrana basar.

## 📚 Her Modülün Amacı

- **Tokenization & Embedding**: Metni sayısal vektörlere dönüştürme.
- **Positional Encoding**: Sıra bilgisini vektörlere ekleme.
- **Attention**: Tokenler arası ilişkileri öğrenme ve görselleştirme.
- **Multi-Head Attention**: Farklı alt uzaylarda paralel dikkat hesaplama.
- **Layer Normalization**: Eğitimde istikrar ve hız.
- **Feed-Forward Network**: Her pozisyona bağımsız doğrusal dönüşüm.
- **Encoder Layer**: Tüm bileşenlerin birleşimiyle tam bir Transformer bloğu.
- **Output Head**: Model çıktısını kelime olasılıklarına çevirme.

## 🎯 Hedef

Bu repo, Transformer mimarisinin temel taşlarını derinlemesine anlamak ve uygulamak isteyenler için referans niteliğindedir. Her modül bağımsız olarak çalıştırılabilir ve kolayca incelenebilir.

## 📄 Lisans

MIT

---

Her klasörün kendi README dosyasında daha fazla teknik detay ve kullanım örneği bulabilirsiniz.