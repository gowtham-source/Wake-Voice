"""
Training Pipeline for Wake Word Detection
Includes:
- Learning rate scheduling with warmup
- Mixup augmentation
- Early stopping
- Model checkpointing
- Cross-validation support
- Comprehensive logging
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model import TANMSFF, LightweightTANMSFF, count_parameters
from dataset import create_data_loaders, mixup_data, mixup_criterion


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_preds)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        return (-smooth_labels * log_preds).sum(dim=-1).mean()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_mixup=True, mixup_alpha=0.4):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and random.random() < 0.5:
            # Apply mixup
            data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            
            # For accuracy, use original predictions
            _, predicted = output.max(1)
            correct += (lam * predicted.eq(target_a).sum().float() + 
                       (1 - lam) * predicted.eq(target_b).sum().float()).item()
        else:
            output = model(data)
            loss = criterion(output, target)
            
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(val_loader), 100. * correct / total, all_preds, all_targets


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    model_type: str = "full",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    label_smoothing: float = 0.1,
    patience: int = 15,
    seed: int = 42,
    n_mels: int = 64
):
    """Main training function"""
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, info = create_data_loaders(
        data_dir, batch_size=batch_size, n_mels=n_mels, seed=seed
    )
    
    print(f"Classes: {info['classes']}")
    print(f"Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
    
    # Create model
    num_classes = info['num_classes']
    if model_type == "full":
        model = TANMSFF(n_mels=n_mels, num_classes=num_classes, dropout=0.3)
    else:
        model = LightweightTANMSFF(n_mels=40, num_classes=num_classes, dropout=0.3)
    
    model = model.to(device)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss function
    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, use_mixup, mixup_alpha
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if early_stopping(val_acc):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'info': info,
                    'config': {
                        'model_type': model_type,
                        'n_mels': n_mels,
                        'num_classes': num_classes
                    }
                }, output_path / "best_model.pt")
                print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(output_path / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_targets, test_preds, 
        target_names=info['classes'],
        digits=3
    ))
    
    # Save training history
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_path)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_targets, test_preds, info['classes'], output_path)
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_epochs': epoch,
        'training_time_minutes': total_time / 60,
        'model_parameters': count_parameters(model)
    }
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, history, results


def plot_training_curves(history, output_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wake Word Detection Model")
    parser.add_argument("--data_dir", type=str, default="recordings", help="Path to recordings directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "lightweight"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mels", type=int, default=64)
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_mixup=not args.no_mixup,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        seed=args.seed,
        n_mels=args.n_mels
    )
