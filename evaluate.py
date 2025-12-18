"""
Comprehensive Evaluation Module for Wake Word Detection
Includes:
- K-Fold Cross-Validation
- Per-class metrics
- Error analysis
- Model comparison
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import TANMSFF, LightweightTANMSFF, count_parameters
from dataset import WakeWordDataset, create_data_loaders
from train import train_epoch, validate, set_seed, LabelSmoothingCrossEntropy


def cross_validate(
    data_dir: str,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model_type: str = "full",
    n_mels: int = 64,
    seed: int = 42
):
    """
    Perform k-fold cross-validation
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load full dataset
    full_dataset = WakeWordDataset(data_dir, augment=False, n_mels=n_mels)
    
    # Get labels for stratification
    labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
    indices = np.arange(len(full_dataset))
    
    # K-Fold
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    all_predictions = []
    all_targets = []
    
    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Create datasets for this fold
        train_dataset = WakeWordDataset(data_dir, augment=True, n_mels=n_mels)
        val_dataset = WakeWordDataset(data_dir, augment=False, n_mels=n_mels)
        
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
        
        # Create model
        num_classes = len(full_dataset.classes)
        if model_type == "full":
            model = TANMSFF(n_mels=n_mels, num_classes=num_classes, dropout=0.3)
        else:
            model = LightweightTANMSFF(n_mels=40, num_classes=num_classes, dropout=0.3)
        model = model.to(device)
        
        # Training setup
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, epochs=epochs,
            steps_per_epoch=len(train_loader), pct_start=0.1
        )
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Load best model and get final predictions
        model.load_state_dict(best_model_state)
        _, final_acc, preds, targets = validate(model, val_loader, criterion, device)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': final_acc,
            'predictions': preds,
            'targets': targets
        })
        
        all_predictions.extend(preds)
        all_targets.extend(targets)
        
        print(f"  Fold {fold + 1} Best Accuracy: {final_acc:.2f}%")
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    print(f"Min: {min(accuracies):.2f}%, Max: {max(accuracies):.2f}%")
    
    # Overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(
        all_targets, all_predictions,
        target_names=full_dataset.classes,
        digits=3
    ))
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': accuracies,
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'classes': full_dataset.classes
    }


def error_analysis(
    model_path: str,
    data_dir: str,
    output_dir: str = "analysis"
):
    """
    Perform detailed error analysis
    """
    from inference import WakeWordDetector
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    detector = WakeWordDetector(model_path, confidence_threshold=0.0)
    
    # Load dataset
    dataset = WakeWordDataset(data_dir, augment=False, n_mels=detector.n_mels)
    
    # Collect predictions
    results = []
    for idx in range(len(dataset)):
        mel_spec, true_label = dataset[idx]
        mel_spec = mel_spec.unsqueeze(0).to(detector.device)
        
        with torch.no_grad():
            logits = detector.model(mel_spec)
            probs = torch.softmax(logits, dim=-1)
        
        confidence, pred_label = probs.max(dim=-1)
        
        results.append({
            'true_label': true_label,
            'pred_label': pred_label.item(),
            'confidence': confidence.item(),
            'correct': true_label == pred_label.item(),
            'file': str(dataset.samples[idx][0])
        })
    
    # Analyze errors
    errors = [r for r in results if not r['correct']]
    
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")
    print(f"Accuracy: {100 * (len(results) - len(errors)) / len(results):.2f}%")
    
    # Confusion pairs
    confusion_pairs = {}
    for e in errors:
        true_class = dataset.idx_to_class[e['true_label']]
        pred_class = dataset.idx_to_class[e['pred_label']]
        pair = (true_class, pred_class)
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    print("\nMost confused pairs:")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true_c, pred_c), count in sorted_pairs[:10]:
        print(f"  {true_c} -> {pred_c}: {count} errors")
    
    # Low confidence correct predictions
    correct_low_conf = [r for r in results if r['correct'] and r['confidence'] < 0.7]
    print(f"\nLow confidence correct predictions (<0.7): {len(correct_low_conf)}")
    
    # High confidence errors
    high_conf_errors = [r for r in errors if r['confidence'] > 0.7]
    print(f"High confidence errors (>0.7): {len(high_conf_errors)}")
    
    # Save detailed error report
    error_report = {
        'total_samples': len(results),
        'total_errors': len(errors),
        'accuracy': 100 * (len(results) - len(errors)) / len(results),
        'confusion_pairs': {f"{k[0]}->{k[1]}": v for k, v in sorted_pairs},
        'errors': errors[:50]  # First 50 errors
    }
    
    with open(output_path / "error_analysis.json", 'w') as f:
        json.dump(error_report, f, indent=2)
    
    # Plot confidence distribution
    correct_confs = [r['confidence'] for r in results if r['correct']]
    error_confs = [r['confidence'] for r in results if not r['correct']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(correct_confs, bins=20, alpha=0.7, label='Correct', color='green')
    ax.hist(error_confs, bins=20, alpha=0.7, label='Errors', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    plt.savefig(output_path / "confidence_distribution.png", dpi=150)
    plt.close()
    
    return error_report


def benchmark_models(data_dir: str, output_dir: str = "benchmarks"):
    """
    Benchmark different model configurations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configurations to test
    configs = [
        {"name": "TAN-MSFF-Full", "model_type": "full", "n_mels": 64},
        {"name": "TAN-MSFF-Light", "model_type": "lightweight", "n_mels": 40},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nBenchmarking: {config['name']}")
        
        # Create model
        if config['model_type'] == 'full':
            model = TANMSFF(n_mels=config['n_mels'], num_classes=18)
        else:
            model = LightweightTANMSFF(n_mels=config['n_mels'], num_classes=18)
        
        model = model.to(device)
        model.eval()
        
        # Count parameters
        params = count_parameters(model)
        
        # Measure inference time
        dummy_input = torch.randn(1, config['n_mels'], 100).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start.record()
                with torch.no_grad():
                    _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                start_time = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        
        results.append({
            'name': config['name'],
            'parameters': params,
            'inference_time_ms': avg_time,
            'model_type': config['model_type']
        })
        
        print(f"  Parameters: {params:,}")
        print(f"  Inference time: {avg_time:.2f} ms")
    
    # Save results
    with open(output_path / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Wake Word Detection Model")
    parser.add_argument("--mode", type=str, default="cv", 
                       choices=["cv", "error", "benchmark"])
    parser.add_argument("--data_dir", type=str, default="recordings")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.mode == "cv":
        cross_validate(args.data_dir, n_folds=args.n_folds, epochs=args.epochs)
    elif args.mode == "error":
        error_analysis(args.model, args.data_dir)
    elif args.mode == "benchmark":
        benchmark_models(args.data_dir)
