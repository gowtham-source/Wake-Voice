"""
Fine-tune the pretrained wake word detection model on augmented data.
Uses lower learning rate and fewer epochs to preserve learned features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

from model import TANMSFF
from dataset import WakeWordDataset, create_data_loaders


def load_pretrained_model(checkpoint_path: str, device: torch.device):
    """Load pretrained model and extract architecture info"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    n_mels = config.get("n_mels", 64)
    num_classes = config.get("num_classes", 18)
    
    # Infer architecture from state dict
    state_dict = checkpoint["model_state_dict"]
    
    if "conv_blocks.0.bn.weight" in state_dict:
        ch0 = state_dict["conv_blocks.0.bn.weight"].shape[0]
    else:
        ch0 = 32
    if "conv_blocks.2.bn.weight" in state_dict:
        ch1 = state_dict["conv_blocks.2.bn.weight"].shape[0]
    else:
        ch1 = 64
    if "conv_blocks.4.bn.weight" in state_dict:
        ch2 = state_dict["conv_blocks.4.bn.weight"].shape[0]
    else:
        ch2 = 128
    channels = [ch0, ch1, ch2]
    
    num_attention_layers = sum(1 for k in state_dict.keys() 
                               if k.startswith("attention_blocks.") and k.endswith(".scale"))
    
    # Get class info
    info = checkpoint.get("info", {})
    classes = info.get("classes", [])
    class_to_idx = info.get("class_to_idx", {})
    
    # Create model
    model = TANMSFF(
        n_mels=n_mels,
        num_classes=num_classes,
        channels=channels,
        num_attention_layers=num_attention_layers
    )
    
    model.load_state_dict(state_dict)
    
    print(f"Loaded pretrained model:")
    print(f"  Architecture: channels={channels}, attention_layers={num_attention_layers}")
    print(f"  Classes: {len(classes)}")
    
    return model, n_mels, classes, class_to_idx


def train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Optional mixup
        if mixup_alpha > 0:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            index = torch.randperm(data.size(0)).to(device)
            mixed_data = lam * data + (1 - lam) * data[index]
            
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target[index])
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained model")
    parser.add_argument("--pretrained", type=str, default="checkpoints/best_model.pt",
                        help="Path to pretrained model")
    parser.add_argument("--data_dir", type=str, default="recordings_augmented",
                        help="Directory with augmented data")
    parser.add_argument("--epochs", type=int, default=30, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--output", type=str, default="checkpoints/finetuned_model.pt",
                        help="Output model path")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of conv blocks to freeze (0-5)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    print("\n" + "=" * 60)
    print("LOADING PRETRAINED MODEL")
    print("=" * 60)
    model, n_mels, classes, class_to_idx = load_pretrained_model(args.pretrained, device)
    model = model.to(device)
    
    # Optionally freeze early layers
    if args.freeze_layers > 0:
        print(f"\nFreezing first {args.freeze_layers} conv blocks...")
        for i, block in enumerate(model.conv_blocks):
            if i < args.freeze_layers * 2:  # 2 blocks per stage
                for param in block.parameters():
                    param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    # Create data loaders
    print("\n" + "=" * 60)
    print("LOADING AUGMENTED DATA")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_mels=n_mels,
        augment=True,  # Additional on-the-fly augmentation
        num_workers=0
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Verify classes match
    if dataset_info['classes'] != classes:
        print("\nWARNING: Class mismatch between pretrained model and new data!")
        print(f"  Pretrained: {classes}")
        print(f"  New data: {dataset_info['classes']}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("FINE-TUNING")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    best_val_acc = 0
    best_epoch = 0
    history = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha=0.2
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'n_mels': n_mels,
                    'num_classes': len(classes),
                    'model_type': 'full',
                    'channels': [32, 64, 128],  # From pretrained
                    'num_attention_layers': 1
                },
                'info': {
                    'classes': classes,
                    'class_to_idx': class_to_idx,
                    'idx_to_class': {v: k for k, v in class_to_idx.items()}
                },
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }
            
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_epochs': args.epochs,
        'training_time_minutes': total_time / 60,
        'pretrained_model': args.pretrained,
        'data_dir': args.data_dir
    }
    
    results_path = Path(args.output).parent / "finetune_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    history_path = Path(args.output).parent / "finetune_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nFine-tuning completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
