# Position-wise Feed-Forward Network 🏗️

This project implements the **Feed-Forward Network (FFN)** component of the Transformer architecture from scratch.

### Purpose
In a Transformer block, the Multi-Head Attention layer captures global relationships between tokens, while the FFN layer applies non-linear transformations to each token's representation. It operates "position-wise," meaning the same FFN is applied to each token in the sequence independently.

### Key Details
- **Linear Layer 1:** Expands dimensionality ($d_{model} \to d_{ff}$).
- **Activation:** Uses **ReLU** to introduce non-linearity.
- **Linear Layer 2:** Projects back to the original model dimension ($d_{ff} \to d_{model}$).
- Implemented using purely **NumPy** matrix multiplications.

### Usage
```bash
python main.py