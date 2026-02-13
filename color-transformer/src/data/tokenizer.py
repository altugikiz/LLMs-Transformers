"""
Tokenizer for color-attribute dataset.
Converts words to tokens and vice versa.
"""

from typing import List, Dict, Optional, Union
from collections import Counter
import json
from pathlib import Path
import pandas as pd


class ColorTokenizer:
    """
    Simple tokenizer for color-attribute sentences.
    Builds vocabulary from dataset and provides encode/decode functionality.
    """
    
    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sentence
        self.eos_token = "<EOS>"  # End of sentence
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Will be filled during fit()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def fit(self, sentences: List[str], min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """
        Build vocabulary from list of sentences.
        
        Args:
            sentences: List of text sentences
            min_freq: Minimum frequency for a word to be included
            max_vocab_size: Maximum vocabulary size (optional)
        """
        # Count all words
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_counts.update(words)
        
        # Filter by frequency
        vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Limit vocabulary size if specified
        if max_vocab_size:
            # Reserve space for special tokens
            max_vocab_size -= len(self.special_tokens)
            vocab_words = vocab_words[:max_vocab_size]
        
        # Build vocabulary with special tokens first
        all_words = self.special_tokens + vocab_words
        
        self.word2idx = {word: idx for idx, word in enumerate(all_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"✅ Vocabulary built: {self.vocab_size} tokens")
        print(f"   Special tokens: {self.special_tokens}")
        print(f"   Regular words: {self.vocab_size - len(self.special_tokens)}")
    
    def encode(
        self, 
        sentence: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Convert sentence to token IDs.
        
        Args:
            sentence: Input sentence
            add_special_tokens: Whether to add <BOS> and <EOS>
            max_length: Optional maximum length (truncate if longer)
            
        Returns:
            List of token IDs
        """
        words = sentence.lower().split()
        
        # Convert to IDs, using <UNK> for unknown words
        token_ids = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        
        if add_special_tokens:
            token_ids = [self.word2idx[self.bos_token]] + token_ids + [self.word2idx[self.eos_token]]
        
        if max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to sentence.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded sentence
        """
        tokens = []
        for idx in token_ids:
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special_tokens and word in self.special_tokens:
                continue
            tokens.append(word)
        
        return " ".join(tokens)
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer vocabulary to file."""
        path = Path(path)
        vocab_data = {
            'word2idx': self.word2idx,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Tokenizer saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load tokenizer vocabulary from file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.special_tokens = vocab_data['special_tokens']
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Update special token attributes
        self.pad_token = self.special_tokens[0]
        self.unk_token = self.special_tokens[1]
        self.bos_token = self.special_tokens[2]
        self.eos_token = self.special_tokens[3]
        
        print(f"✅ Tokenizer loaded from {path}")
        print(f"   Vocabulary size: {self.vocab_size}")


# Test the tokenizer
if __name__ == "__main__":
    # Sample sentences
    test_sentences = [
        "red apple sweet",
        "blue sea deep",
        "green forest fresh"
    ]
    
    # Create and fit tokenizer
    tokenizer = ColorTokenizer()
    tokenizer.fit(test_sentences)
    
    # Test encoding/decoding
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {sentence}")
        print(f"Encoded : {encoded}")
        print(f"Decoded : {decoded}")