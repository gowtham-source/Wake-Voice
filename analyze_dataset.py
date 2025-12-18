"""
Dataset Analysis Script for Wake Word Detection
Analyzes audio characteristics to inform model design
"""

import os
import wave
import struct
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_wav(filepath):
    """Analyze a single WAV file"""
    try:
        with wave.open(str(filepath), 'rb') as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / framerate
            
            # Read audio data
            frames = wav.readframes(n_frames)
            if sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32)
            else:
                audio = np.frombuffer(frames, dtype=np.uint8)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
            
            return {
                'channels': n_channels,
                'sample_width': sample_width,
                'sample_rate': framerate,
                'duration': duration,
                'n_frames': n_frames,
                'rms': rms,
                'max_amplitude': np.max(np.abs(audio)),
                'valid': True
            }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def main():
    recordings_dir = Path("recordings")
    
    stats = defaultdict(list)
    class_counts = {}
    all_durations = []
    all_sample_rates = []
    
    print("=" * 60)
    print("WAKE WORD DATASET ANALYSIS")
    print("=" * 60)
    
    # Analyze each class
    for class_dir in sorted(recordings_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            wav_files = list(class_dir.glob("*.wav"))
            class_counts[class_name] = len(wav_files)
            
            class_durations = []
            for wav_file in wav_files:
                info = analyze_wav(wav_file)
                if info['valid']:
                    class_durations.append(info['duration'])
                    all_durations.append(info['duration'])
                    all_sample_rates.append(info['sample_rate'])
                    stats[class_name].append(info)
            
            if class_durations:
                print(f"\n{class_name}:")
                print(f"  Samples: {len(wav_files)}")
                print(f"  Duration: {np.mean(class_durations):.2f}s (±{np.std(class_durations):.2f}s)")
                print(f"  Range: {np.min(class_durations):.2f}s - {np.max(class_durations):.2f}s")
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total samples: {sum(class_counts.values())}")
    print(f"Samples per class: {list(class_counts.values())[0]} (balanced)")
    
    if all_durations:
        print(f"\nAudio Duration:")
        print(f"  Mean: {np.mean(all_durations):.3f}s")
        print(f"  Std: {np.std(all_durations):.3f}s")
        print(f"  Min: {np.min(all_durations):.3f}s")
        print(f"  Max: {np.max(all_durations):.3f}s")
    
    if all_sample_rates:
        print(f"\nSample Rate: {all_sample_rates[0]} Hz")
    
    # Get sample info from first valid file
    for class_name, class_stats in stats.items():
        if class_stats:
            sample = class_stats[0]
            print(f"\nAudio Format:")
            print(f"  Channels: {sample['channels']}")
            print(f"  Sample Width: {sample['sample_width']} bytes ({sample['sample_width']*8} bits)")
            print(f"  Sample Rate: {sample['sample_rate']} Hz")
            break
    
    print("\n" + "=" * 60)
    print("DATASET CHARACTERISTICS FOR MODEL DESIGN")
    print("=" * 60)
    print(f"\n- Small dataset ({sum(class_counts.values())} samples) -> Need data augmentation")
    print(f"- Multi-class ({len(class_counts)} classes) -> Classification task")
    print(f"- Short utterances (~{np.mean(all_durations):.1f}s) -> Lightweight model suitable")
    print(f"- Single speaker likely -> May need augmentation for generalization")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDED APPROACH")
    print("=" * 60)
    print("""
1. Feature Extraction: Mel-spectrograms or MFCCs
2. Architecture: Lightweight CNN + Attention or Temporal Convolutions
3. Data Augmentation: Time stretch, pitch shift, noise injection, SpecAugment
4. Training: Cross-validation due to small dataset
5. Regularization: Dropout, weight decay, mixup
""")

if __name__ == "__main__":
    main()
