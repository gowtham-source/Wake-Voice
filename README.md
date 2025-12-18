# Wake Word Detection System

A novel **TAN-MSFF (Temporal Attention Network with Multi-Scale Feature Fusion)** architecture for audio wake word/phrase detection, optimized for small datasets.

## Dataset

Harry Potter spells dataset:
- **18 classes**: accio, arresto-momentum, avada-kedavra, bombarda, confringo, crucio, depulso, descendo, diffindo, expelliarmus, flipendo, glacius, imperio, incendio, levioso, lumos, reparo, tranformation
- **30 samples per class** (540 total)
- **Audio format**: 16kHz, mono, 16-bit WAV
- **Average duration**: ~0.8 seconds

## Architecture: TAN-MSFF

### Key Innovations

1. **Multi-Scale Temporal Convolutions**: Captures phonetic patterns at different granularities using parallel convolutions with kernel sizes 1, 3, 5, and 7.

2. **Squeeze-and-Excitation Attention**: Channel recalibration to emphasize important frequency bands.

3. **Temporal Self-Attention**: Captures long-range dependencies in speech with relative positional encoding.

4. **Residual Connections with Learnable Scaling**: Stable training with adaptive residual weighting.

5. **Statistics Pooling**: Captures both mean and variance of temporal features for robust representation.

### Model Variants

| Model | Parameters | Use Case |
|-------|------------|----------|
| TAN-MSFF Full | ~500K | High accuracy |
| TAN-MSFF Lightweight | ~100K | Edge deployment |

## Installation

```bash
# Using uv (recommended)
uv add torch torchaudio librosa scikit-learn matplotlib numpy
```

## Usage

### Training

```bash
# Full model training
uv run train.py --data_dir recordings --epochs 100 --batch_size 32

# Lightweight model
uv run train.py --data_dir recordings --model_type lightweight --epochs 100
```

### Inference

```bash
# Single file inference
uv run inference.py --model checkpoints/best_model.pt --audio path/to/audio.wav
```

### Evaluation

```bash
# Cross-validation
uv run evaluate.py --mode cv --data_dir recordings --n_folds 5

# Error analysis
uv run evaluate.py --mode error --model checkpoints/best_model.pt

# Benchmark models
uv run evaluate.py --mode benchmark
```

## Training Features

- **Data Augmentation**: Time stretch, pitch shift, noise injection, time shift, volume change
- **SpecAugment**: Frequency and time masking on spectrograms
- **Mixup**: Sample mixing for regularization
- **Label Smoothing**: Prevents overconfident predictions
- **OneCycleLR**: Learning rate scheduling with warmup
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stable training

## Project Structure

```
wake_voice/
├── recordings/           # Dataset directory
│   ├── accio/
│   ├── arresto-momentum/
│   └── ...
├── model.py             # TAN-MSFF architecture
├── dataset.py           # Dataset and augmentation
├── train.py             # Training pipeline
├── inference.py         # Inference module
├── evaluate.py          # Evaluation and cross-validation
├── analyze_dataset.py   # Dataset analysis
└── checkpoints/         # Saved models
```

## Performance Optimization Tips

1. **Small Dataset**: Use aggressive augmentation and cross-validation
2. **Overfitting**: Increase dropout, use mixup and label smoothing
3. **Underfitting**: Reduce regularization, increase model capacity
4. **Fast Inference**: Use lightweight model variant

## API Usage

```python
from inference import WakeWordDetector

# Load model
detector = WakeWordDetector("checkpoints/best_model.pt")

# Predict from file
result = detector.predict_file("audio.wav")
print(f"Detected: {result['predicted_class']} ({result['confidence']:.2f})")

# Predict from waveform
import numpy as np
waveform = np.random.randn(16000)  # 1 second of audio
result = detector.predict_waveform(waveform, sample_rate=16000)
```

## Streaming Detection

```python
from inference import StreamingDetector

detector = StreamingDetector(
    "checkpoints/best_model.pt",
    window_duration=1.5,
    confidence_threshold=0.7
)

# Process audio chunks
while True:
    chunk = get_audio_chunk()  # Your audio source
    result = detector.process_chunk(chunk)
    if result:
        print(f"Wake word detected: {result['predicted_class']}")
```
