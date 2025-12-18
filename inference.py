"""
Inference Module for Wake Word Detection
Supports:
- Single file inference
- Real-time streaming inference
- Batch inference
- Confidence thresholding
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import time

from model import TANMSFF, LightweightTANMSFF


class WakeWordDetector:
    """
    Wake Word Detection inference class
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (auto-detected if None)
            confidence_threshold: Minimum confidence for detection
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self._load_model(model_path)
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.max_duration = 2.0
        self.max_samples = int(self.max_duration * self.sample_rate)
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=self.n_mels,
            power=2.0
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model configuration
        config = checkpoint.get('config', {})
        self.n_mels = config.get('n_mels', 64)
        num_classes = config.get('num_classes', 18)
        model_type = config.get('model_type', 'full')
        
        # Get class mappings
        info = checkpoint.get('info', {})
        self.classes = info.get('classes', [])
        self.class_to_idx = info.get('class_to_idx', {})
        self.idx_to_class = info.get('idx_to_class', {})
        
        # Convert idx_to_class keys to integers if they're strings
        if self.idx_to_class and isinstance(list(self.idx_to_class.keys())[0], str):
            self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}
        
        # Create model
        if model_type == 'full':
            self.model = TANMSFF(n_mels=self.n_mels, num_classes=num_classes)
        else:
            self.model = LightweightTANMSFF(n_mels=40, num_classes=num_classes)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {self.classes}")
        print(f"Device: {self.device}")
    
    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio waveform"""
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Pad or truncate
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad_length = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def _to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        waveform = waveform.to(self.device)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def predict_file(self, audio_path: str) -> Dict:
        """
        Predict wake word from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Preprocess
        waveform = self._preprocess_audio(waveform, sample_rate)
        
        # Convert to mel spectrogram
        mel_spec = self._to_mel_spectrogram(waveform)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(mel_spec)
            probs = torch.softmax(logits, dim=-1)
        inference_time = time.time() - start_time
        
        # Get prediction
        confidence, predicted_idx = probs.max(dim=-1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        predicted_class = self.idx_to_class.get(predicted_idx, f"class_{predicted_idx}")
        
        # Get top-k predictions
        top_k = min(5, len(self.classes))
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        top_predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            class_name = self.idx_to_class.get(int(idx), f"class_{idx}")
            top_predictions.append({
                'class': class_name,
                'confidence': float(prob)
            })
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_detected': confidence >= self.confidence_threshold,
            'top_predictions': top_predictions,
            'inference_time_ms': inference_time * 1000
        }
    
    def predict_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Predict wake word from numpy waveform
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Preprocess
        waveform = self._preprocess_audio(waveform, sample_rate)
        
        # Convert to mel spectrogram
        mel_spec = self._to_mel_spectrogram(waveform)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(mel_spec)
            probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        confidence, predicted_idx = probs.max(dim=-1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        predicted_class = self.idx_to_class.get(predicted_idx, f"class_{predicted_idx}")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_detected': confidence >= self.confidence_threshold
        }
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Batch prediction for multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict_file(path)
                result['file'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': path,
                    'error': str(e)
                })
        return results


class StreamingDetector:
    """
    Real-time streaming wake word detector
    Uses a sliding window approach
    """
    
    def __init__(
        self,
        model_path: str,
        window_duration: float = 1.5,
        hop_duration: float = 0.5,
        confidence_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize streaming detector
        
        Args:
            model_path: Path to trained model
            window_duration: Duration of detection window in seconds
            hop_duration: Hop between windows in seconds
            confidence_threshold: Minimum confidence for detection
            device: Device for inference
        """
        self.detector = WakeWordDetector(
            model_path, device, confidence_threshold
        )
        
        self.sample_rate = 16000
        self.window_samples = int(window_duration * self.sample_rate)
        self.hop_samples = int(hop_duration * self.sample_rate)
        
        # Buffer for streaming audio
        self.buffer = np.zeros(0, dtype=np.float32)
        
        # Detection state
        self.last_detection = None
        self.detection_cooldown = 1.0  # seconds
        self.last_detection_time = 0
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process an audio chunk and return detection if found
        
        Args:
            audio_chunk: Audio samples as numpy array
            
        Returns:
            Detection result or None
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Check if we have enough samples
        if len(self.buffer) < self.window_samples:
            return None
        
        # Get window
        window = self.buffer[:self.window_samples]
        
        # Run detection
        result = self.detector.predict_waveform(window, self.sample_rate)
        
        # Update buffer (slide by hop)
        self.buffer = self.buffer[self.hop_samples:]
        
        # Check cooldown
        current_time = time.time()
        if result['is_detected']:
            if current_time - self.last_detection_time > self.detection_cooldown:
                self.last_detection_time = current_time
                self.last_detection = result
                return result
        
        return None
    
    def reset(self):
        """Reset the streaming buffer"""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_detection = None
        self.last_detection_time = 0


def demo_inference(model_path: str, audio_path: str):
    """Demo inference on a single file"""
    detector = WakeWordDetector(model_path)
    result = detector.predict_file(audio_path)
    
    print("\n" + "=" * 50)
    print("INFERENCE RESULT")
    print("=" * 50)
    print(f"File: {audio_path}")
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Detected: {result['is_detected']}")
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print("\nTop predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wake Word Detection Inference")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    demo_inference(args.model, args.audio)
