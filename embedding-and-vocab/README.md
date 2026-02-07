# Vocabulary & Embedding: From IDs to Meaning 🌐

This project implements the crucial link between raw token IDs and high-dimensional semantic vectors.

### Components
1. **VocabManager:**
   - Handles the mapping between strings and integers.
   - Includes special tokens like `<PAD>` and `<UNK>` for robust processing.
2. **EmbeddingLayer:**
   - A trainable lookup table that converts discrete IDs into continuous vectors.
   - Demonstrates how semantic space is initialized before deep learning begins.

### Implementation Logic
The workflow transforms `text -> tokens -> IDs -> embeddings`. Once in embedding form, data is ready to be processed by **Transformer Layers**.

### Usage
```bash
python main.py