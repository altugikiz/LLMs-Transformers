# Micro-Attention: Neural Relationship Visualizer 🧠

This project is a "from-scratch" implementation of the **Scaled Dot-Product Attention** mechanism, the core building block of Transformers and Modern Large Language Models (LLMs). 

Developed as part of my **LLM Course** studies, this repository demonstrates the mathematical intuition behind how machines learn relationships between data points (tokens) without relying on heavy deep learning frameworks like PyTorch or TensorFlow.

## 🚀 Overview

The goal of this project is to implement the $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ formula using only **NumPy**. 

It simulates:
- **Query (Q)**, **Key (K)**, and **Value (V)** matrix interactions.
- **Score Scaling** to prevent gradient vanishing/exploding issues.
- **Attention Map Visualization** using Matplotlib to see which words "focus" on each other.

## 🛠️ Tech Stack
- **Python 3.x**
- **NumPy**: For matrix operations and linear algebra.
- **Matplotlib**: For generating attention heatmaps.

## 📂 Project Structure
- `attention_engine.py`: Contains the `ScaledDotProductAttention` class logic.
- `main.py`: A demonstration script that processes a sample sentence and generates a heatmap.
- `requirements.txt`: Project dependencies.

## 📊 How It Works
The engine takes word embeddings, calculates the dot product between Queries and Keys, applies a Softmax to get a probability distribution, and finally weights the Values. The resulting heatmap shows the "Attention Scores" between tokens.

## ⚡ Quick Start
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/micro-attention.git](https://github.com/yourusername/micro-attention.git)