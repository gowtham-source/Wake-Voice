"""
Dataset and Data Augmentation for Wake Word Detection
Includes comprehensive audio augmentation strategies for small datasets
"""

import os
import random
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Lazy import torchaudio transforms to avoid DLL issues
_torchaudio_transforms = None

def get_torchaudio_transforms():
    global _torchaudio_transforms
    if _torchaudio_transforms is None:
        import torchaudio.transforms as T
        _torchaudio_transforms = T
    return _torchaudio_transforms


class AudioAugmentation:
    """
    Comprehensive audio augmentation pipeline for wake word detection.
    Designed to maximize diversity from limited samples.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        p_time_stretch: float = 0.3,
        p_pitch_shift: float = 0.3,
        p_add_noise: float = 0.4,
        p_time_mask: float = 0.3,
        p_freq_mask: float = 0.3,
        p_volume: float = 0.3,
        p_time_shift: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.p_time_stretch = p_time_stretch
        self.p_pitch_shift = p_pitch_shift
        self.p_add_noise = p_add_noise
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_volume = p_volume
        self.p_time_shift = p_time_shift

    def time_stretch(self, waveform: torch.Tensor, rate: float = None) -> torch.Tensor:
        """Time stretch without changing pitch"""
        if rate is None:
            rate = random.uniform(0.85, 1.15)

        # Use torchaudio's stretch
        T = get_torchaudio_transforms()
        stretch = T.TimeStretch(n_freq=201, fixed_rate=rate)
        # Convert to spectrogram, stretch, convert back
        spec = torch.stft(waveform, n_fft=400, return_complex=True)
        stretched = stretch(spec.unsqueeze(0)).squeeze(0)
        return torch.istft(stretched, n_fft=400, length=int(waveform.shape[-1] * rate))

    def pitch_shift(self, waveform: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """Shift pitch by n semitones using simple resampling"""
        if n_steps is None:
            n_steps = random.randint(-2, 2)  # Reduced range

        if n_steps == 0:
            return waveform

        # Simple pitch shift by changing playback rate
        ratio = 2 ** (n_steps / 12)
        orig_len = waveform.shape[-1]
        
        # Resample using interpolation (memory efficient)
        new_len = int(orig_len / ratio)
        if new_len < 10:
            return waveform
            
        # Use linear interpolation
        indices = torch.linspace(0, orig_len - 1, new_len)
        indices_floor = indices.long().clamp(0, orig_len - 2)
        frac = indices - indices_floor.float()
        
        if waveform.dim() == 2:
            resampled = waveform[:, indices_floor] * (1 - frac) + waveform[:, indices_floor + 1] * frac
        else:
            resampled = waveform[indices_floor] * (1 - frac) + waveform[indices_floor + 1] * frac

        # Adjust length back to original
        if resampled.shape[-1] > orig_len:
            return resampled[..., :orig_len]
        else:
            pad = orig_len - resampled.shape[-1]
            return torch.nn.functional.pad(resampled, (0, pad))

    def add_noise(self, waveform: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """Add Gaussian noise at specified SNR"""
        if snr_db is None:
            snr_db = random.uniform(10, 30)

        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def time_shift(
        self, waveform: torch.Tensor, shift_ratio: float = None
    ) -> torch.Tensor:
        """Shift audio in time (circular)"""
        if shift_ratio is None:
            shift_ratio = random.uniform(-0.2, 0.2)

        shift = int(waveform.shape[-1] * shift_ratio)
        return torch.roll(waveform, shift, dims=-1)

    def volume_change(
        self, waveform: torch.Tensor, gain_db: float = None
    ) -> torch.Tensor:
        """Change volume"""
        if gain_db is None:
            gain_db = random.uniform(-6, 6)

        gain = 10 ** (gain_db / 20)
        return waveform * gain

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations"""
        if random.random() < self.p_volume:
            waveform = self.volume_change(waveform)

        if random.random() < self.p_time_shift:
            waveform = self.time_shift(waveform)

        if random.random() < self.p_pitch_shift:
            waveform = self.pitch_shift(waveform)

        if random.random() < self.p_add_noise:
            waveform = self.add_noise(waveform)

        return waveform


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    Applied on mel-spectrograms
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5,
    ):
        T = get_torchaudio_transforms()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return spec

        for _ in range(self.n_freq_masks):
            spec = self.freq_mask(spec)

        for _ in range(self.n_time_masks):
            spec = self.time_mask(spec)

        return spec


class WakeWordDataset(Dataset):
    """
    Dataset for wake word detection with comprehensive augmentation
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 2.0,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        augment: bool = True,
        spec_augment: bool = True,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.n_mels = n_mels
        self.augment = augment

        # Mel spectrogram transform
        T = get_torchaudio_transforms()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Augmentations
        self.audio_augment = AudioAugmentation(sample_rate) if augment else None
        self.spec_augment = SpecAugment() if spec_augment and augment else None

        # Load file paths and labels
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for audio_file in class_dir.glob("*.wav"):
                self.samples.append((audio_file, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        data, sr = sf.read(str(path))
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if necessary
        if sr != self.sample_rate:
            T = get_torchaudio_transforms()
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to fixed length
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad_length = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform.squeeze(0)

    def _to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db.squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # Load audio
        waveform = self._load_audio(path)

        # Apply audio augmentation
        if self.audio_augment is not None and self.augment:
            waveform = self.audio_augment(waveform)

        # Convert to mel spectrogram
        mel_spec = self._to_mel_spectrogram(waveform)

        # Apply SpecAugment
        if self.spec_augment is not None and self.augment:
            mel_spec = self.spec_augment(mel_spec)

        return mel_spec, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        class_counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1

        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.classes)
        return weights


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders with stratified split
    """
    from sklearn.model_selection import train_test_split

    # Create full dataset without augmentation first to get all samples
    # Remove 'augment' from kwargs if present to avoid duplicate
    dataset_kwargs.pop('augment', None)
    full_dataset = WakeWordDataset(data_dir, augment=False, **dataset_kwargs)

    # Get indices and labels for stratified split
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=labels, random_state=seed
    )

    # Second split: train vs val
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),
        stratify=train_val_labels,
        random_state=seed,
    )

    # Create datasets
    train_dataset = WakeWordDataset(data_dir, augment=True, **dataset_kwargs)
    val_dataset = WakeWordDataset(data_dir, augment=False, **dataset_kwargs)
    test_dataset = WakeWordDataset(data_dir, augment=False, **dataset_kwargs)

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(test_dataset, test_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "num_classes": len(full_dataset.classes),
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "idx_to_class": full_dataset.idx_to_class,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    # Test dataset
    dataset = WakeWordDataset("recordings", augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class to idx: {dataset.class_to_idx}")

    # Test a sample
    mel_spec, label = dataset[0]
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")

    # Test data loaders
    train_loader, val_loader, test_loader, info = create_data_loaders(
        "recordings", batch_size=16
    )
    print(
        f"\nData split: Train={info['train_size']}, Val={info['val_size']}, Test={info['test_size']}"
    )

    # Test batch
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}")
        print(f"Labels shape: {batch_y.shape}")
        break
