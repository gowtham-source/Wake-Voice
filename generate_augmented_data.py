"""
Generate augmented audio data for fine-tuning the wake word detection model.
Creates realistic variations to improve real-time microphone performance.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import random
from typing import List, Tuple
import argparse
from tqdm import tqdm


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file"""
    data, sr = sf.read(str(path))
    return data, sr


def save_audio(path: Path, data: np.ndarray, sr: int):
    """Save audio file"""
    sf.write(str(path), data, sr)


# Augmentation functions
def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add white noise"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def add_colored_noise(audio: np.ndarray, noise_level: float = 0.008) -> np.ndarray:
    """Add pink/brown noise (more realistic ambient noise)"""
    # Generate pink noise using cumulative sum of white noise
    white = np.random.randn(len(audio))
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    pink = pink / (np.std(pink) + 1e-8) * noise_level
    return audio + pink


def change_volume(audio: np.ndarray, factor_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Random volume change"""
    factor = random.uniform(*factor_range)
    return audio * factor


def time_shift(audio: np.ndarray, sr: int, max_shift_ms: int = 200) -> np.ndarray:
    """Shift audio in time (add silence at start or end)"""
    max_shift = int(sr * max_shift_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    
    if shift > 0:
        # Add silence at start
        return np.concatenate([np.zeros(shift), audio[:-shift]])
    elif shift < 0:
        # Add silence at end
        return np.concatenate([audio[-shift:], np.zeros(-shift)])
    return audio


def add_reverb(audio: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    """Simple reverb effect"""
    # Create impulse response
    reverb_length = int(sr * 0.1)  # 100ms reverb
    impulse = np.exp(-np.linspace(0, 5, reverb_length)) * decay
    
    # Convolve
    reverbed = np.convolve(audio, impulse, mode='same')
    return audio * 0.7 + reverbed * 0.3


def pitch_shift_simple(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Simple pitch shift by resampling (changes duration slightly)"""
    factor = 2 ** (semitones / 12)
    indices = np.arange(0, len(audio), factor)
    indices = indices[indices < len(audio)].astype(int)
    shifted = audio[indices]
    
    # Pad or trim to original length
    if len(shifted) < len(audio):
        shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
    else:
        shifted = shifted[:len(audio)]
    
    return shifted


def speed_change(audio: np.ndarray, factor_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Change speed (and pitch) slightly"""
    factor = random.uniform(*factor_range)
    indices = np.arange(0, len(audio), factor)
    indices = indices[indices < len(audio)].astype(int)
    changed = audio[indices]
    
    # Pad or trim to original length
    if len(changed) < len(audio):
        changed = np.pad(changed, (0, len(audio) - len(changed)))
    else:
        changed = changed[:len(audio)]
    
    return changed


def add_room_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate room acoustics with early reflections"""
    # Early reflections at different delays
    delays_ms = [10, 20, 35, 50]
    gains = [0.6, 0.4, 0.3, 0.2]
    
    result = audio.copy()
    for delay_ms, gain in zip(delays_ms, gains):
        delay_samples = int(sr * delay_ms / 1000)
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * gain
            result = result + delayed
    
    # Normalize
    result = result / (np.max(np.abs(result)) + 1e-8) * np.max(np.abs(audio))
    return result


def low_pass_filter(audio: np.ndarray, cutoff_ratio: float = 0.8) -> np.ndarray:
    """Simple low-pass filter (simulates phone/distance)"""
    # Simple moving average filter
    window_size = max(2, int(1 / cutoff_ratio))
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(audio, kernel, mode='same')
    return filtered


def add_click_noise(audio: np.ndarray, num_clicks: int = 3) -> np.ndarray:
    """Add random click/pop artifacts"""
    result = audio.copy()
    for _ in range(num_clicks):
        pos = random.randint(0, len(audio) - 1)
        intensity = random.uniform(0.1, 0.3) * np.max(np.abs(audio))
        result[pos] += intensity * random.choice([-1, 1])
    return result


def generate_augmentations(audio: np.ndarray, sr: int, num_augmentations: int = 5) -> List[Tuple[np.ndarray, str]]:
    """Generate multiple augmented versions of an audio sample"""
    augmented = []
    
    # Define augmentation pipelines for realistic mic variations
    pipelines = [
        # Pipeline 1: Noisy environment
        lambda x: add_colored_noise(add_noise(x, 0.003), 0.005),
        
        # Pipeline 2: Volume variation + noise
        lambda x: add_noise(change_volume(x, (0.6, 1.4)), 0.004),
        
        # Pipeline 3: Room acoustics
        lambda x: add_room_effect(add_noise(x, 0.002), sr),
        
        # Pipeline 4: Distance effect (quieter + reverb)
        lambda x: add_reverb(change_volume(x, (0.4, 0.7)), sr, 0.4),
        
        # Pipeline 5: Slight speed variation + noise
        lambda x: add_noise(speed_change(x, (0.92, 1.08)), 0.003),
        
        # Pipeline 6: Time shifted + noise
        lambda x: add_colored_noise(time_shift(x, sr, 150), 0.004),
        
        # Pipeline 7: Phone quality (low pass + noise)
        lambda x: add_noise(low_pass_filter(x, 0.7), 0.005),
        
        # Pipeline 8: Pitch variation
        lambda x: add_noise(pitch_shift_simple(x, sr, random.uniform(-1, 1)), 0.003),
        
        # Pipeline 9: Heavy noise environment
        lambda x: add_colored_noise(add_noise(x, 0.008), 0.01),
        
        # Pipeline 10: Click artifacts + room
        lambda x: add_room_effect(add_click_noise(x, 2), sr),
    ]
    
    # Select random pipelines
    selected = random.sample(pipelines, min(num_augmentations, len(pipelines)))
    
    for i, pipeline in enumerate(selected):
        try:
            aug_audio = pipeline(audio)
            # Normalize to prevent clipping
            max_val = np.max(np.abs(aug_audio))
            if max_val > 0.95:
                aug_audio = aug_audio * 0.9 / max_val
            augmented.append((aug_audio, f"aug{i+1}"))
        except Exception as e:
            print(f"Augmentation failed: {e}")
            continue
    
    return augmented


def main():
    parser = argparse.ArgumentParser(description="Generate augmented audio data")
    parser.add_argument("--input_dir", type=str, default="recordings", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="recordings_augmented", help="Output directory")
    parser.add_argument("--num_augmentations", type=int, default=5, help="Number of augmentations per sample")
    parser.add_argument("--include_original", action="store_true", default=True, help="Include original samples")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("GENERATING AUGMENTED DATA")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per sample: {args.num_augmentations}")
    print()
    
    # Process each class
    total_generated = 0
    
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        wav_files = list(class_dir.glob("*.wav"))
        print(f"\nProcessing {class_name}: {len(wav_files)} files")
        
        class_count = 0
        
        for wav_file in tqdm(wav_files, desc=f"  {class_name}"):
            audio, sr = load_audio(wav_file)
            
            # Save original if requested
            if args.include_original:
                orig_path = output_class_dir / f"{wav_file.stem}_orig.wav"
                save_audio(orig_path, audio, sr)
                class_count += 1
            
            # Generate augmentations
            augmented = generate_augmentations(audio, sr, args.num_augmentations)
            
            for aug_audio, aug_name in augmented:
                aug_path = output_class_dir / f"{wav_file.stem}_{aug_name}.wav"
                save_audio(aug_path, aug_audio, sr)
                class_count += 1
        
        total_generated += class_count
        print(f"  Generated {class_count} files for {class_name}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_generated} audio files generated")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
