# Multi-Head Attention: Parallel Neural Context ⚡

This repository contains a low-level implementation of the **Multi-Head Attention** mechanism, as described in the "Attention Is All You Need" paper. 

### Why Multi-Head?
While standard attention focuses on a single relationship, **Multi-Head Attention** allows the model to jointly attend to information from different representation subspaces. In this project:
- The input is split into **N independent heads**.
- Each head calculates its own attention scores.
- Results are concatenated and projected back.

### Key Features
- **Head Splitting:** Mathematical reshaping of matrices for parallel processing.
- **Batched Operations:** Optimized for multi-sample inputs.
- **Visualization:** Comparative heatmaps for each attention head.

### Usage
```bash
python main.py