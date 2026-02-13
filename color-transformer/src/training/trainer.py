"""
Trainer class for transformer model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import time
from typing import Dict, Optional, Any

from src.training.loss import CrossEntropyLoss, LabelSmoothingLoss
from src.training.metrics import compute_perplexity, compute_accuracy

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Run 'pip install wandb' to enable logging.")


class Trainer:
    """
    Trainer for ColorTransformer model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        use_wandb: bool = False
    ):
        """
        Args:
            model: ColorTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer instance
            config: Training configuration dictionary
            device: Device to train on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if use_wandb and not WANDB_AVAILABLE:
            print("⚠️ wandb requested but not installed. Continuing without wandb.")
        
        # Loss function
        if config.get('label_smoothing', 0) > 0:
            self.criterion = LabelSmoothingLoss(
                smoothing=config['label_smoothing'],
                ignore_index=-100
            )
        else:
            self.criterion = CrossEntropyLoss(ignore_index=-100)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 10)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create experiment directory
        self.exp_dir = Path(f"experiments/exp_{time.strftime('%Y%m%d_%H%M%S')}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n🚀 Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Experiment: {self.exp_dir}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Optimizer: AdamW (lr={config.get('learning_rate', 1e-4)})")
        if self.use_wandb:
            print(f"  Weights & Biases: enabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        # Model'in tüm parametrelerini kontrol et
        if self.current_epoch == 0:
            print("\n🔍 Model parameter stats:")
            for name, param in self.model.named_parameters():
                if param.data is not None:
                    print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")
                    if torch.isnan(param.data).any():
                        print(f"    ⚠️ NaN in {name}!")
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Her batch'te NaN kontrolü
            if batch_idx == 0:
                print(f"\n📥 Batch {batch_idx}:")
                print(f"  input_ids: {input_ids.shape}, min={input_ids.min()}, max={input_ids.max()}")
                print(f"  labels: {labels.shape}, min={labels.min()}, max={labels.max()}")
                print(f"  attention_mask: {attention_mask.shape}, sum={attention_mask.sum()}")
            
            # Forward pass
            try:
                logits = self.model(input_ids, attention_mask)
            except Exception as e:
                print(f"❌ Forward pass error: {e}")
                raise
            
            # Loss hesapla (BU SATIRI EKLE!)
            loss = self.criterion(logits, labels)
            
            # Logits kontrolü
            if torch.isnan(logits).any():
                print(f"\n❌ NaN detected in logits at batch {batch_idx}!")
                print(f"  logits shape: {logits.shape}")
                print(f"  logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
                
                # Model'in ara katmanlarını kontrol etmek için hook ekle
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                        print(f"  NaN in {module.__class__.__name__}")
                        print(f"    output shape: {output.shape}")
                        print(f"    output stats: min={output.min()}, max={output.max()}")
                
                # Her katmana hook ekle
                hooks = []
                for name, module in self.model.named_modules():
                    hooks.append(module.register_forward_hook(hook_fn))
                
                # Tekrar forward yap
                logits = self.model(input_ids, attention_mask)
                
                # Hook'ları temizle
                for hook in hooks:
                    hook.remove()
                
                raise ValueError("NaN in logits - training stopped")
            
            # Backward pass (BU SATIRLARI EKLE!)
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (BU SATIRI EKLE!)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update metrics (BU SATIRLARI EKLE!)
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += (labels != -100).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        # Ortalama loss'u hesapla (BU SATIRI EKLE!)
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Compute accuracy
                predictions = logits.argmax(dim=-1)
                mask = (labels != -100)
                correct = (predictions == labels) & mask
                
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += mask.sum().item()
                total_correct += correct.sum().item()
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            'accuracy': accuracy
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/perplexity': metrics['perplexity'],
                'val/accuracy': accuracy,
                'epoch': self.current_epoch
            })
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  🏆 New best model saved! (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\n{'='*50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*50}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            if train_metrics is None:  # Bu kontrolü ekle
                train_metrics = {'loss': 0.0, 'perplexity': 0.0}
            
            # Validate
            val_metrics = self.validate()
            
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
            print(f"  Train - loss: {train_metrics['loss']:.4f}, ppl: {train_metrics['perplexity']:.2f}")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, ppl: {val_metrics['perplexity']:.2f}, acc: {val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.config.get('early_stopping_patience'):
                # TODO: Implement early stopping
                pass
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {self.checkpoint_dir / 'best.pt'}")
        print(f"{'='*50}")


# Test trainer
if __name__ == "__main__":
    print("🧪 Testing Trainer...")
    
    # This is just a placeholder - actual training will be done in scripts/train.py
    print("\n✅ Trainer ready for training!")