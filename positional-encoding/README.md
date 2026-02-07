# Positional Encoding From Scratch 📈

Since Transformer models do not use recurrence or convolution, they have no inherent sense of the order of words in a sequence. This project implements **Positional Encoding** using Sine and Cosine functions to inject "positional information" into word embeddings.

### Why Sine and Cosine?
The authors of "Attention Is All You Need" chose these functions because:
1. They allow the model to easily learn to attend by **relative positions**.
2. They scale to **longer sequences** than those seen during training.
3. Each dimension of the positional encoding corresponds to a sinusoid.

### Implementation
- Generates a unique vector for each position in a sequence.
- Uses alternating **Sine** (for even indices) and **Cosine** (for odd indices).
- Visualizes the unique "fingerprint" of positions using heatmaps.

### Usage
```bash
python main.py