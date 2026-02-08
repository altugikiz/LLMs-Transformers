from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Model ve Tokenizer'ı indir (Hugging Face Hub'dan otomatik gelir)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Giriş metnini hazırla
prompt = "The future of Artificial Intelligence is"
# Senin yazdığın Tokenizer ve VocabManager mantığı burada devreye giriyor:
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 3. Modelin metin üretmesini sağla (Inference)
# Senin yazdığın Output Head ve Softmax mantığı burada milyarlarca parametreyle çalışıyor
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# 4. Sayısal ID'leri tekrar metne çevir (Decoding)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Giriş: {prompt}")
print(f"Modelin Devamı: {generated_text}")