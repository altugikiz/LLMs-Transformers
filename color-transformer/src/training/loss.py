"""
Loss functions for transformer training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for language modeling.
    Ignores padding tokens (where label = -100).
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len) with -100 for padding
            
        Returns:
            Loss value
        """
        # Reshape logits for cross entropy
        # (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)


class LabelSmoothingLoss(nn.Module):
    """
    Cross entropy with label smoothing.
    Helps prevent overfitting and improves generalization.
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len) with -100 for padding
            
        Returns:
            Smoothed loss value
        """
        vocab_size = logits.size(-1)
        
        # Reshape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
            # Mask out padding tokens
            mask = (labels == self.ignore_index).unsqueeze(1)
            true_dist = true_dist.masked_fill(mask, 0)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        # Mask out padding
        mask = (labels != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


# Test loss functions
if __name__ == "__main__":
    print("🧪 Testing loss functions...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    # Create random logits and labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Set some padding tokens
    labels[:, 5:] = -100
    
    # Test CrossEntropyLoss
    ce_loss = CrossEntropyLoss()
    loss1 = ce_loss(logits, labels)
    print(f"\nCrossEntropyLoss: {loss1.item():.4f}")
    
    # Test LabelSmoothingLoss
    smooth_loss = LabelSmoothingLoss(smoothing=0.1)
    loss2 = smooth_loss(logits, labels)
    print(f"LabelSmoothingLoss: {loss2.item():.4f}")
    
    print("\n✅ Loss functions ready!")