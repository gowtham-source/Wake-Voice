"""
Inference Module for Wake Word Detection
Supports:
- Single file inference
- Real-time streaming inference from microphone
- Batch inference
- Confidence thresholding
"""

import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import time
import sys
import queue
import threading

from model import TANMSFF, LightweightTANMSFF

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class VoiceActivityDetector:
    """
    Fast energy-based Voice Activity Detection (VAD)
    Uses RMS energy and zero-crossing rate to detect speech
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.025,  # 25ms frames
        energy_threshold: float = 0.01,  # Initial energy threshold
        zcr_threshold: float = 0.01,  # Zero-crossing rate threshold
        adaptation_rate: float = 0.95,  # Background noise adaptation rate
        min_speech_frames: int = 3,  # Minimum consecutive frames for speech
    ):
        """
        Initialize VAD
        
        Args:
            sample_rate: Audio sample rate
            frame_duration: Duration of each frame in seconds
            energy_threshold: RMS energy threshold (adaptive)
            zcr_threshold: Zero-crossing rate threshold
            adaptation_rate: Rate at which background noise adapts (0-1)
            min_speech_frames: Minimum frames of speech before triggering
        """
        self.sample_rate = sample_rate
        self.frame_samples = int(frame_duration * sample_rate)
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.adaptation_rate = adaptation_rate
        self.min_speech_frames = min_speech_frames
        
        # Background noise estimation
        self.background_energy = energy_threshold
        self.background_zcr = zcr_threshold
        
        # Speech state tracking
        self.speech_frames = 0
        self.is_speech = False
        
    def calculate_rms_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio frame"""
        if len(audio) == 0:
            return 0.0
        return np.sqrt(np.mean(audio ** 2))
    
    def calculate_zcr(self, audio: np.ndarray) -> float:
        """Calculate zero-crossing rate"""
        if len(audio) < 2:
            return 0.0
        # Count sign changes
        sign_changes = np.sum(np.diff(np.signbit(audio)) != 0)
        return sign_changes / len(audio)
    
    def detect(self, audio: np.ndarray) -> bool:
        """
        Detect if audio contains voice activity
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            True if voice activity detected, False otherwise
        """
        if len(audio) < self.frame_samples:
            return False
        
        # Process in frames
        has_speech = False
        for i in range(0, len(audio) - self.frame_samples + 1, self.frame_samples):
            frame = audio[i:i + self.frame_samples]
            
            # Calculate features
            energy = self.calculate_rms_energy(frame)
            zcr = self.calculate_zcr(frame)
            
            # Update background noise (exponential moving average)
            if energy < self.background_energy * 2:  # Only update if not speech
                self.background_energy = (
                    self.adaptation_rate * self.background_energy + 
                    (1 - self.adaptation_rate) * energy
                )
                self.background_zcr = (
                    self.adaptation_rate * self.background_zcr + 
                    (1 - self.adaptation_rate) * zcr
                )
            
            # Adaptive threshold (background + margin)
            adaptive_threshold = self.background_energy * 2.5
            
            # Check if frame contains speech
            # Speech typically has higher energy and moderate ZCR
            is_speech_frame = (
                energy > adaptive_threshold and
                zcr > self.zcr_threshold and
                zcr < 0.5  # Too high ZCR indicates noise
            )
            
            if is_speech_frame:
                self.speech_frames += 1
                if self.speech_frames >= self.min_speech_frames:
                    has_speech = True
                    self.is_speech = True
            else:
                # Reset counter if no speech detected
                if self.speech_frames > 0:
                    self.speech_frames = max(0, self.speech_frames - 1)
                if self.speech_frames == 0:
                    self.is_speech = False
        
        return has_speech
    
    def reset(self):
        """Reset VAD state"""
        self.speech_frames = 0
        self.is_speech = False


class WakeWordDetector:
    """
    Wake Word Detection inference class
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
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
            power=2.0,
        ).to(self.device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def _load_model(self, model_path: str):
        """Load the trained model"""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Get model configuration
        config = checkpoint.get("config", {})
        self.n_mels = config.get("n_mels", 64)
        num_classes = config.get("num_classes", 18)
        model_type = config.get("model_type", "full")
        
        # Get architecture params from config or infer from state dict
        channels = config.get("channels", None)
        num_attention_layers = config.get("num_attention_layers", None)
        
        # Infer architecture from state dict if not in config
        state_dict = checkpoint["model_state_dict"]
        if channels is None:
            # Infer from conv_blocks weights
            # conv_blocks.0 is first MultiScaleConvBlock, output channels = bn.weight size
            if "conv_blocks.0.bn.weight" in state_dict:
                ch0 = state_dict["conv_blocks.0.bn.weight"].shape[0]
            else:
                ch0 = 64
            if "conv_blocks.2.bn.weight" in state_dict:
                ch1 = state_dict["conv_blocks.2.bn.weight"].shape[0]
            else:
                ch1 = 128
            if "conv_blocks.4.bn.weight" in state_dict:
                ch2 = state_dict["conv_blocks.4.bn.weight"].shape[0]
            else:
                ch2 = 256
            channels = [ch0, ch1, ch2]
        
        if num_attention_layers is None:
            # Count attention blocks
            num_attention_layers = sum(1 for k in state_dict.keys() if k.startswith("attention_blocks.") and k.endswith(".scale"))

        # Get class mappings
        info = checkpoint.get("info", {})
        self.classes = info.get("classes", [])
        self.class_to_idx = info.get("class_to_idx", {})
        self.idx_to_class = info.get("idx_to_class", {})

        # Convert idx_to_class keys to integers if they're strings
        if self.idx_to_class and isinstance(list(self.idx_to_class.keys())[0], str):
            self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}

        # Create model with inferred architecture
        if model_type == "full":
            self.model = TANMSFF(
                n_mels=self.n_mels, 
                num_classes=num_classes,
                channels=channels,
                num_attention_layers=num_attention_layers
            )
        else:
            self.model = LightweightTANMSFF(n_mels=40, num_classes=num_classes)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Architecture: channels={channels}, attention_layers={num_attention_layers}")
        print(f"Classes: {self.classes}")
        print(f"Device: {self.device}")

    def _preprocess_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
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
            waveform = waveform[:, : self.max_samples]
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
        # Load audio using soundfile (same as training)
        data, sample_rate = sf.read(str(audio_path))
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

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
            top_predictions.append({"class": class_name, "confidence": float(prob)})

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_detected": confidence >= self.confidence_threshold,
            "top_predictions": top_predictions,
            "inference_time_ms": inference_time * 1000,
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
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_detected": confidence >= self.confidence_threshold,
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
                result["file"] = path
                results.append(result)
            except Exception as e:
                results.append({"file": path, "error": str(e)})
        return results


class StreamingDetector:
    """
    Real-time streaming wake word detector
    Uses a sliding window approach with Voice Activity Detection
    """

    def __init__(
        self,
        model_path: str,
        window_duration: float = 1.5,
        hop_duration: float = 0.5,
        confidence_threshold: float = 0.7,
        device: Optional[str] = None,
        use_vad: bool = True,
        vad_energy_threshold: float = 0.01,
    ):
        """
        Initialize streaming detector

        Args:
            model_path: Path to trained model
            window_duration: Duration of detection window in seconds
            hop_duration: Hop between windows in seconds
            confidence_threshold: Minimum confidence for detection
            device: Device for inference
            use_vad: Enable Voice Activity Detection
            vad_energy_threshold: VAD energy threshold
        """
        self.detector = WakeWordDetector(model_path, device, confidence_threshold)

        self.sample_rate = 16000
        self.window_samples = int(window_duration * self.sample_rate)
        self.hop_samples = int(hop_duration * self.sample_rate)

        # Voice Activity Detection
        self.use_vad = use_vad
        if use_vad:
            self.vad = VoiceActivityDetector(
                sample_rate=self.sample_rate,
                energy_threshold=vad_energy_threshold,
            )
        else:
            self.vad = None

        # Buffer for streaming audio
        self.buffer = np.zeros(0, dtype=np.float32)

        # Detection state
        self.last_detection = None
        self.detection_cooldown = 1.0  # seconds
        self.last_detection_time = 0
        
        # VAD statistics
        self.vad_rejected_count = 0
        self.vad_accepted_count = 0

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
        window = self.buffer[: self.window_samples]

        # Voice Activity Detection - only process if speech detected
        if self.use_vad and self.vad is not None:
            if not self.vad.detect(window):
                # No voice activity, skip inference
                self.vad_rejected_count += 1
                # Still update buffer to keep sliding
                self.buffer = self.buffer[self.hop_samples :]
                return None
            
            self.vad_accepted_count += 1

        # Run detection only on voice activity
        result = self.detector.predict_waveform(window, self.sample_rate)

        # Update buffer (slide by hop)
        self.buffer = self.buffer[self.hop_samples :]

        # Check cooldown
        current_time = time.time()
        if result["is_detected"]:
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
        if self.vad is not None:
            self.vad.reset()
        self.vad_rejected_count = 0
        self.vad_accepted_count = 0
    
    def get_vad_stats(self) -> Dict:
        """Get VAD statistics"""
        total = self.vad_rejected_count + self.vad_accepted_count
        if total == 0:
            return {"rejected": 0, "accepted": 0, "rejection_rate": 0.0}
        return {
            "rejected": self.vad_rejected_count,
            "accepted": self.vad_accepted_count,
            "rejection_rate": self.vad_rejected_count / total,
        }


def demo_inference(model_path: str, audio_path: str, threshold: float = 0.5):
    """Demo inference on a single file"""
    detector = WakeWordDetector(model_path, confidence_threshold=threshold)
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
    for pred in result["top_predictions"]:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")

    return result


def stream_from_microphone(
    model_path: str,
    threshold: float = 0.7,
    chunk_duration: float = 0.5,
    window_duration: float = 1.5,
    hop_duration: float = 0.5,
    device: Optional[str] = None,
    use_vad: bool = True,
    vad_energy_threshold: float = 0.01,
):
    """
    Stream audio from microphone and perform real-time wake word detection
    
    Args:
        model_path: Path to trained model
        threshold: Confidence threshold for detection
        chunk_duration: Duration of audio chunks to read from mic (seconds)
        window_duration: Duration of detection window (seconds)
        hop_duration: Hop between detection windows (seconds)
        device: Device for inference (auto-detected if None)
        use_vad: Enable Voice Activity Detection to filter noise
        vad_energy_threshold: VAD energy threshold (lower = more sensitive)
    """
    if not HAS_SOUNDDEVICE:
        print("ERROR: sounddevice is required for microphone streaming.")
        print("Install it with: uv add sounddevice")
        sys.exit(1)
    
    # Initialize streaming detector
    print("Initializing streaming detector...")
    detector = StreamingDetector(
        model_path=model_path,
        window_duration=window_duration,
        hop_duration=hop_duration,
        confidence_threshold=threshold,
        device=device,
        use_vad=use_vad,
        vad_energy_threshold=vad_energy_threshold,
    )
    
    sample_rate = detector.sample_rate
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Queue for audio chunks (non-blocking, prevents overflow)
    audio_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    
    print("\n" + "=" * 50)
    print("MICROPHONE STREAMING MODE")
    print("=" * 50)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk duration: {chunk_duration} s")
    print(f"Window duration: {window_duration} s")
    print(f"Confidence threshold: {threshold}")
    print(f"Voice Activity Detection: {'Enabled' if use_vad else 'Disabled'}")
    if use_vad:
        print(f"VAD energy threshold: {vad_energy_threshold}")
    print("\nListening for wake words... (Press Ctrl+C to stop)")
    print("-" * 50)
    
    def audio_callback(indata, frames, time_info, status):
        """Fast callback function - just queues audio data"""
        if status:
            if status.input_overflow:
                # Silently drop old data if queue is full to prevent overflow
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
            else:
                print(f"Audio status: {status}", file=sys.stderr)
        
        # Convert to 1D numpy array (mono) and copy to avoid reference issues
        audio_chunk = indata.flatten().copy().astype(np.float32)
        
        # Non-blocking queue put - drop chunk if queue is full
        try:
            audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            # Queue full, drop this chunk to prevent overflow
            pass
    
    def process_audio_chunks():
        """Process audio chunks in separate thread"""
        while not stop_event.is_set():
            try:
                # Get chunk with timeout to allow checking stop_event
                audio_chunk = audio_queue.get(timeout=0.1)
                
                # Process chunk
                result = detector.process_chunk(audio_chunk)
                
                if result:
                    print(f"\n🎯 WAKE WORD DETECTED!")
                    print(f"   Class: {result['predicted_class']}")
                    print(f"   Confidence: {result['confidence']:.4f}")
                    print(f"   Time: {time.strftime('%H:%M:%S')}")
                    print("-" * 50)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}", file=sys.stderr)
    
    try:
        # Start processing thread
        process_thread = threading.Thread(target=process_audio_chunks, daemon=True)
        process_thread.start()
        
        # Start audio stream
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback,
            latency='low',  # Lower latency
        ):
            print("Streaming started. Speak into your microphone...")
            while True:
                sd.sleep(100)
                
    except KeyboardInterrupt:
        print("\n\nStopping microphone stream...")
        stop_event.set()
        if use_vad:
            stats = detector.get_vad_stats()
            print(f"\nVAD Statistics:")
            print(f"  Accepted chunks: {stats['accepted']}")
            print(f"  Rejected chunks: {stats['rejected']}")
            print(f"  Rejection rate: {stats['rejection_rate']:.1%}")
        detector.reset()
        print("Stream stopped.")
    except Exception as e:
        print(f"\nError during streaming: {e}", file=sys.stderr)
        stop_event.set()
        detector.reset()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wake Word Detection Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on audio file
  python inference.py --audio path/to/audio.wav
  
  # Real-time microphone streaming
  python inference.py --mic
  
  # Microphone streaming with custom threshold
  python inference.py --mic --threshold 0.8
        """
    )
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--audio", type=str, help="Path to audio file (required if --mic not used)")
    parser.add_argument("--mic", action="store_true", help="Stream from microphone (real-time inference)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for detection")
    parser.add_argument("--chunk-duration", type=float, default=0.5, help="Audio chunk duration for mic streaming (seconds)")
    parser.add_argument("--window-duration", type=float, default=1.5, help="Detection window duration (seconds)")
    parser.add_argument("--hop-duration", type=float, default=0.5, help="Hop duration between windows (seconds)")
    parser.add_argument("--no-vad", action="store_true", help="Disable Voice Activity Detection")
    parser.add_argument("--vad-threshold", type=float, default=0.01, help="VAD energy threshold (lower = more sensitive)")

    args = parser.parse_args()

    # Validate arguments
    if not args.mic and not args.audio:
        parser.error("Either --audio or --mic must be specified")

    if args.mic:
        # Microphone streaming mode
        stream_from_microphone(
            model_path=args.model,
            threshold=args.threshold,
            chunk_duration=args.chunk_duration,
            window_duration=args.window_duration,
            hop_duration=args.hop_duration,
            use_vad=not args.no_vad,
            vad_energy_threshold=args.vad_threshold,
        )
    else:
        # File inference mode
        demo_inference(args.model, args.audio, threshold=args.threshold)
