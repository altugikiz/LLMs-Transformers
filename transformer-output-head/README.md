# Transformer Output Head: From Vectors to Words 🗣️

The final stage of a Transformer model. This project implements the mechanism that converts high-dimensional hidden states back into human-readable word probabilities.

### Components
- **Linear Transformation:** Maps the `d_model` dimension back to the `vocab_size`.
- **Softmax Activation:** Converts raw scores (logits) into a probability distribution where the sum equals 1.
- **Greedy Decoding:** Demonstrates the simplest form of word selection by picking the highest probability.

### Mathematical Flow
`Hidden States (16) -> Linear Layer -> Logits (9) -> Softmax -> Probabilities (9) -> Word`

### Usage
```bash
python main.py