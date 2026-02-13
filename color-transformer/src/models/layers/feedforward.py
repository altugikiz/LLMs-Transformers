"""
Position-wise Feed-Forward Network (FFN) for Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension (usually 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights (Xavier/Glorot initialization)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # FFN(x) = dropout(ReLU(linear1(x))) -> linear2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class GELUFeedForward(nn.Module):
    """
    Feed-Forward Network with GELU activation (used in GPT models).
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


# Test FeedForward
if __name__ == "__main__":
    print("🧪 Testing FeedForward layers...")
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048  # Usually 4 * d_model
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test standard FFN
    ffn = FeedForward(d_model, d_ff)
    output = ffn(x)
    print(f"\nStandard FeedForward:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {output.norm().item():.4f}")
    
    # Test GELU FFN
    gelu_ffn = GELUFeedForward(d_model, d_ff)
    gelu_output = gelu_ffn(x)
    print(f"\nGELU FeedForward:")
    print(f"  Output shape: {gelu_output.shape}")
    print(f"  Output norm: {gelu_output.norm().item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n📊 FeedForward parameters: {total_params:,}")