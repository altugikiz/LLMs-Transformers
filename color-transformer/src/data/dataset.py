"""
PyTorch Dataset for color-attribute transformer project.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np

from src.data.tokenizer import ColorTokenizer


class ColorSentenceDataset(Dataset):
    """
    PyTorch Dataset for color-attribute sentences.
    Each sample is a sentence that starts with a color and ends with an attribute.
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: Optional[ColorTokenizer] = None,
        split: str = 'train',
        max_length: int = 20,
        train_ratio: float = 0.8
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV file with sentences
            tokenizer: Tokenizer instance (if None, creates new one)
            split: 'train', 'val', or 'test'
            max_length: Maximum sequence length
            train_ratio: Ratio of data to use for training
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.split = split
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Create or use tokenizer
        if tokenizer is None:
            self.tokenizer = ColorTokenizer()
            self.tokenizer.fit(self.df['sentence'].tolist())
        else:
            self.tokenizer = tokenizer
        
        # Split data
        self._create_splits(train_ratio)
        
        print(f"✅ {split} set loaded: {len(self)} samples")
    
    def _create_splits(self, train_ratio: float):
        """Create train/val/test splits."""
        n = len(self.df)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * (1 - train_ratio) / 2)
        
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with:
                - 'input_ids': Token IDs for the sentence (with <BOS>)
                - 'labels': Token IDs for prediction (with <EOS>)
                - 'attention_mask': Mask for padding
                - 'color': Original color (for analysis)
                - 'attribute': Original attribute (for analysis)
        """
        # Get actual index
        actual_idx = self.indices[idx]
        row = self.df.iloc[actual_idx]
        
        # Encode sentence
        tokens = self.tokenizer.encode(
            row['sentence'],
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        # Create input_ids (all tokens except last) and labels (all tokens except first)
        # This is for autoregressive training
        input_ids = tokens[:-1]  # Remove last token
        labels = tokens[1:]      # Remove first token, shift by 1
        
        # Pad sequences to max_length
        input_ids = self._pad_sequence(input_ids, self.max_length - 1)
        labels = self._pad_sequence(labels, self.max_length - 1, pad_value=-100)  # -100 is ignored in loss
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != self.tokenizer.word2idx[self.tokenizer.pad_token] else 0 
                         for token in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'color': row['color'],
            'attribute': row['attribute']
        }
    
    def _pad_sequence(self, sequence: List[int], target_length: int, pad_value: int = 0) -> List[int]:
        """Pad or truncate sequence to target length."""
        if len(sequence) > target_length:
            return sequence[:target_length]
        else:
            return sequence + [pad_value] * (target_length - len(sequence))
    
    def get_tokenizer(self) -> ColorTokenizer:
        """Return the tokenizer instance."""
        return self.tokenizer


def create_dataloaders(
    data_path: Path,
    batch_size: int = 32,
    max_length: int = 20,
    train_ratio: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, ColorTokenizer]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_path: Path to CSV file
        batch_size: Batch size for training
        max_length: Maximum sequence length
        train_ratio: Ratio for training split
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)

    Automatically detects device and sets pin_memory accordingly.
    """
    # First create tokenizer using all data
    temp_dataset = ColorSentenceDataset(
        data_path=data_path,
        split='train',  # Just to initialize
        max_length=max_length,
        train_ratio=train_ratio
    )
    tokenizer = temp_dataset.get_tokenizer()
    
    # Create datasets with the same tokenizer
    train_dataset = ColorSentenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split='train',
        max_length=max_length,
        train_ratio=train_ratio
    )
    
    val_dataset = ColorSentenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split='val',
        max_length=max_length,
        train_ratio=train_ratio
    )
    
    test_dataset = ColorSentenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split='test',
        max_length=max_length,
        train_ratio=train_ratio
    )
    
    # Detect device and set pin_memory accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # MPS doesn't support pin_memory yet
    use_pin_memory = device.type == 'cuda'
    
    if device.type == 'mps':
        print("ℹ️  MPS device detected: disabling pin_memory")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Device: {device.type}")
    print(f"   Pin memory: {use_pin_memory}")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    return train_loader, val_loader, test_loader, tokenizer


# Test the dataset
if __name__ == "__main__":
    # Create a small test dataset first
    from src.data.generator import ColorDataGenerator
    
    print("Creating test data...")
    generator = ColorDataGenerator(seed=42)
    test_df = generator.generate_dataset(num_samples=100)
    test_path = Path("data/raw/test_data.csv")
    test_df.to_csv(test_path, index=False)
    
    # Test dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        data_path=test_path,
        batch_size=16,
        max_length=15
    )
    
    # Test one batch
    print("\n📦 Testing one batch:")
    batch = next(iter(train_loader))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    
    # Show device info
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"\n💻 Using device: {device.type}")
    
    # Show sample
    print("\n📝 Sample from batch:")
    idx = 0
    print(f"Input tokens: {batch['input_ids'][idx].tolist()}")
    print(f"Decoded input: {tokenizer.decode(batch['input_ids'][idx].tolist())}")
    print(f"Labels: {batch['labels'][idx].tolist()}")
    print(f"Color: {batch['color'][idx]}")
    print(f"Attribute: {batch['attribute'][idx]}")