from transformers import pipeline

# 1. Pipeline oluşturma
# 'sentiment-analysis' (duygu analizi) görevini belirtiyoruz.
# Hugging Face, bu görev için en uygun modeli (genelde DistilBERT) otomatik indirir.
classifier = pipeline("sentiment-analysis")

# 2. Test metinleri
texts = [
    "I am very happy to learn how transformers work from scratch!",
    "I am confused about the math, but the code is working fine.",
    "This error is very frustrating and I hate it."
]

# 3. Analiz ve Sonuçları Yazdırma
print("--- Sentiment Analysis Results ---")
results = classifier(texts)

for text, result in zip(texts, results):
    label = result['label']
    score = result['score']
    print(f"\nText: {text}")
    print(f"Prediction: {label} (Confidence: {score:.2%})")