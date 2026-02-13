#!/usr/bin/env python
"""
Training script for ColorTransformer model.
"""

import torch
import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.generator import ColorDataGenerator
from src.data.dataset import create_dataloaders
from src.models.transformer import ColorTransformer
from src.training.trainer import Trainer


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n📋 Configuration loaded from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"💻 Using device: {device}")
    
    # Generate data if needed
    data_path = Path(config['data']['path'])
    if not data_path.exists() or args.regenerate_data:
        print("\n📊 Generating synthetic dataset...")
        generator = ColorDataGenerator(seed=42)
        df = generator.generate_dataset(
            num_samples=config['data']['num_samples'],
            output_path=data_path
        )
        print(f"   Dataset saved to {data_path}")
    
    # Create dataloaders
    print("\n📚 Creating dataloaders...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        data_path=data_path,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        train_ratio=config['data']['train_ratio'],
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    print("\n🤖 Creating model...")
    model = ColorTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['data']['max_length'],
        dropout=config['model']['dropout'],
        pad_token_id=tokenizer.word2idx[tokenizer.pad_token]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config['training'],
        device=device,
        use_wandb=args.use_wandb
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))
    
    # Start training
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ColorTransformer model")
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/training_config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--regenerate-data',
        action='store_true',
        help='Regenerate synthetic dataset even if it exists'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    
    args = parser.parse_args()
    main(args)