# Layer Normalization From Scratch ⚖️

This project implements **Layer Normalization**, a critical technique used in Transformers (like GPT and BERT) to stabilize the training of deep neural networks.

### Why Layer Normalization?
Unlike Batch Normalization, LayerNorm normalizes the inputs across the features for each training example independently. This makes it ideal for:
- **Recurrent networks** and **Transformers**.
- Tasks with **variable sequence lengths**.
- Training with **small batch sizes**.

### Implementation Details
- **Mean & Variance calculation** across the feature dimension ($d_{model}$).
- **Epsilon ($\epsilon$)** handling for numerical stability.
- **Learnable parameters ($\gamma, \beta$)** support for scale and shift.

### Usage
```bash
python main.py