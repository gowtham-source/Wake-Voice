"""Comprehensive test script for inference across all classes"""
from inference import WakeWordDetector
from pathlib import Path
import random

# Load detector
detector = WakeWordDetector('checkpoints/best_model.pt')

print("\n" + "=" * 60)
print("COMPREHENSIVE INFERENCE TEST - ALL CLASSES")
print("=" * 60)

recordings_dir = Path('recordings')
class_results = {}
total_correct = 0
total_samples = 0

for class_dir in sorted(recordings_dir.iterdir()):
    if class_dir.is_dir():
        class_name = class_dir.name
        wav_files = list(class_dir.glob('*.wav'))
        
        correct = 0
        confidences = []
        errors = []
        
        for wav_file in wav_files:
            result = detector.predict_file(str(wav_file))
            predicted = result['predicted_class']
            confidence = result['confidence']
            
            if predicted == class_name:
                correct += 1
                confidences.append(confidence)
            else:
                errors.append((wav_file.name, predicted, confidence))
        
        accuracy = 100 * correct / len(wav_files) if wav_files else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        class_results[class_name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(wav_files),
            'avg_confidence': avg_conf,
            'errors': errors
        }
        
        total_correct += correct
        total_samples += len(wav_files)
        
        status = "✓" if accuracy == 100 else "⚠" if accuracy >= 80 else "✗"
        print(f"{status} {class_name:20s}: {correct:2d}/{len(wav_files):2d} ({accuracy:5.1f}%) | Avg conf: {avg_conf:.3f}")
        
        # Show errors if any
        if errors:
            for fname, pred, conf in errors[:2]:  # Show max 2 errors
                print(f"     ✗ {fname}: predicted '{pred}' ({conf:.3f})")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
overall_accuracy = 100 * total_correct / total_samples if total_samples else 0
print(f"Overall Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.2f}%")

# Classes with errors
error_classes = [c for c, r in class_results.items() if r['accuracy'] < 100]
if error_classes:
    print(f"\nClasses with errors: {', '.join(error_classes)}")
