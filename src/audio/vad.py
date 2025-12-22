# src/audio/vad.py
"""
Voice Activity Detection using Silero VAD.
Silero VAD offers better accuracy than WebRTC VAD, especially for Romanian.
"""
from __future__ import annotations
import warnings
import torch
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*")

# Global model cache (loaded once)
_silero_model = None
_silero_utils = None


def _load_silero():
    """Load Silero VAD model (cached globally)."""
    global _silero_model, _silero_utils
    if _silero_model is None:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            verbose=False
        )
        _silero_model = model
        _silero_utils = utils
    return _silero_model, _silero_utils


class VAD:
    """
    Voice Activity Detection using Silero VAD.
    
    Drop-in replacement for WebRTC VAD with same interface.
    """
    
    def __init__(self, sample_rate: int, aggressiveness: int = 2, frame_ms: int = 30):
        """
        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000)
            aggressiveness: 0-3, higher = more aggressive (maps to threshold)
            frame_ms: Frame duration (10, 20, or 30 ms) - for compatibility
        """
        assert sample_rate in (8000, 16000), "Silero VAD requires 8000 or 16000 Hz"
        self.sr = sample_rate
        self.frame_ms = frame_ms
        
        # Map aggressiveness to threshold (0=permissive, 3=strict)
        # Lower threshold = more speech detected
        threshold_map = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6}
        self.threshold = threshold_map.get(aggressiveness, 0.5)
        
        # Load model
        self.model, _ = _load_silero()
        self.model.reset_states()
        
        # Calculate expected samples per frame
        self.samples_per_frame = int(self.sr * frame_ms / 1000)
    
    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Check if audio frame contains speech.
        
        Args:
            pcm_bytes: Raw PCM audio bytes (int16 little-endian)
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Convert bytes to numpy array (int16 -> float32)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Silero expects tensor
            audio_tensor = torch.from_numpy(audio_float)
            
            # Get speech probability
            speech_prob = self.model(audio_tensor, self.sr).item()
            
            return speech_prob >= self.threshold
            
        except Exception:
            # Fallback: assume no speech on error
            return False
    
    def reset(self):
        """Reset model states (call between utterances)."""
        self.model.reset_states()


class WebRTCVAD:
    """
    Legacy WebRTC VAD wrapper for backward compatibility.
    Use VAD class (Silero) for better accuracy.
    """
    
    def __init__(self, sample_rate: int, aggressiveness: int = 2, frame_ms: int = 30):
        import webrtcvad
        assert frame_ms in (10, 20, 30), "WebRTC VAD frame must be 10/20/30 ms"
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(aggressiveness)
    
    def is_speech(self, pcm_bytes: bytes) -> bool:
        return self.vad.is_speech(pcm_bytes, self.sr)
