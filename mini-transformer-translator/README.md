# Mini Transformer Translator

A minimal, educational implementation of a Transformer-based sequence-to-sequence translation model. This project demonstrates the core components of a Transformer, including tokenization, training, and inference, using a simple text corpus.

## Directory Structure

- `main.py` : Entry point for training and evaluation.
- `requirements.txt` : Python dependencies.
- `data/corpus.txt` : Parallel text corpus for training/testing.
- `engine/`
  - `tokenizer.py` : Simple tokenizer and detokenizer.
  - `trainer.py` : Training loop and utilities.
  - `transformer.py` : Transformer model architecture.

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Replace or extend `data/corpus.txt` with your own parallel data.

## Usage

To train and evaluate the model, run:
```
python main.py
```

## Requirements

- Python 3.7+
- numpy
- matplotlib
- torch
- tqdm

## Features

- Minimal, readable Transformer implementation
- Customizable tokenizer
- Training and evaluation routines
- Easily extensible codebase

## Training Results

After training, the script will output loss and accuracy metrics per epoch. Example output:

```
Epoch 1/10
Train Loss: 2.31 | Train Accuracy: 34.2%
Validation Loss: 2.10 | Validation Accuracy: 38.7%

Epoch 2/10
Train Loss: 1.85 | Train Accuracy: 45.1%
Validation Loss: 1.78 | Validation Accuracy: 47.9%
...
```

You can visualize the training progress using the generated matplotlib plots.

## Notes

- This project is for educational purposes and uses a small dataset for demonstration.
- For larger datasets or production use, further optimizations and improvements are recommended.

## License

MIT