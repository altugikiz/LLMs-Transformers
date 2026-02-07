# Tokenization Fundamentals: BPE From Scratch 🔤

This project explores how Large Language Models convert raw text into a format they can process. It focuses on the **Byte Pair Encoding (BPE)** algorithm, the standard for GPT-style models.

### Why Tokenization?
Neural networks can't "read" text. Tokenization:
- Breaks text into manageable chunks (Tokens).
- Maps each token to a unique **Vocabulary ID**.
- Uses **Subword Merging** to handle rare words and reduce vocabulary size.

### Learning Points
- Initializing with UTF-8 byte encoding.
- Calculating pair frequencies.
- Iterative merging to create a compact vocabulary.