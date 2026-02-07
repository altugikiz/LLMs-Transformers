# Transformer Encoder Layer From Scratch 🚀

This is the final stage of the fundamental series. I have integrated all previous components—**Multi-Head Attention**, **Layer Normalization**, **Positional Encoding**, and **Feed-Forward Networks**—into a fully functional **Transformer Encoder Block**.

### Features
- **Residual Connections:** Implements "Add & Norm" to prevent vanishing gradients.
- **Modularity:** Every component is decoupled and reusable.
- **Pure NumPy:** No high-level AI frameworks used, demonstrating deep mathematical understanding.

### Architecture Flow
1. **Input + Positional Encoding**
2. **Multi-Head Self-Attention**
3. **Add & Layer Normalization**
4. **Position-wise Feed-Forward**
5. **Add & Layer Normalization**

### Usage
```bash
python main.py