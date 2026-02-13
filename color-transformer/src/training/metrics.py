"""
Metrics for evaluating transformer model performance.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute perplexity (exponential of cross entropy loss).
    Lower is better.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for padding
        
    Returns:
        Perplexity value
    """
    # Reshape
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')
    
    # Perplexity = exp(loss)
    return torch.exp(loss).item()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute token-level accuracy.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for padding
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Create mask for non-padding tokens
    mask = (labels != -100)
    
    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return {
        'accuracy': accuracy.item(),
        'correct_tokens': correct.sum().item(),
        'total_tokens': mask.sum().item()
    }


def compute_color_attribute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    color_tokens: List[int],
    attribute_tokens: List[int]
) -> Dict[str, float]:
    """
    Special metric for color-attribute task.
    Checks if model correctly predicts the attribute for given color.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        tokenizer: ColorTokenizer instance
        color_tokens: List of token IDs for colors
        attribute_tokens: List of token IDs for attributes
        
    Returns:
        Dictionary with color-attribute accuracy
    """
    predictions = logits.argmax(dim=-1)
    
    # Find positions of colors in input
    color_positions = []
    attribute_positions = []
    
    for b in range(labels.size(0)):
        for pos in range(labels.size(1)):
            if labels[b, pos].item() in color_tokens:
                color_positions.append((b, pos))
            if labels[b, pos].item() in attribute_tokens:
                attribute_positions.append((b, pos))
    
    # Check if predicted attribute matches expected
    correct_attributes = 0
    total_attributes = len(attribute_positions)
    
    for b, pos in attribute_positions:
        if predictions[b, pos].item() == labels[b, pos].item():
            correct_attributes += 1
    
    return {
        'color_attribute_accuracy': correct_attributes / total_attributes if total_attributes > 0 else 0.0,
        'correct_attributes': correct_attributes,
        'total_attributes': total_attributes
    }


# Test metrics
if __name__ == "__main__":
    print("🧪 Testing metrics...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    # Create random logits and labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, 5:] = -100
    
    # Test perplexity
    ppl = compute_perplexity(logits, labels)
    print(f"\nPerplexity: {ppl:.2f}")
    
    # Test accuracy
    acc = compute_accuracy(logits, labels)
    print(f"Accuracy: {acc['accuracy']:.4f}")
    print(f"  Correct: {acc['correct_tokens']}/{acc['total_tokens']}")
    
    print("\n✅ Metrics ready!")