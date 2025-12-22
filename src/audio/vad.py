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
from collections import deque

warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*")

# Global model cache (loaded once)
_silero_model = None


def _load_silero():
    """Load Silero VAD model (cached globally)."""
    global _silero_model
    if _silero_model is None:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            verbose=False
        )
        _silero_model = model
    return _silero_model


class VAD:
    """
    Voice Activity Detection using Silero VAD.
    
    Drop-in replacement for WebRTC VAD with same interface.
    Silero VAD requires minimum 512 samples (32ms at 16kHz), so we buffer.
    """
    
    def __init__(self, sample_rate: int, aggressiveness: int = 2, frame_ms: int = 30):
        """
        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000)
            aggressiveness: 0-3, higher = more aggressive (maps to threshold)
            frame_ms: Frame duration (10, 20, or 30 ms) - for compatibility
        """
        if sample_rate not in (8000, 16000):
            # Fallback to WebRTC VAD for unsupported sample rates
            self._use_webrtc = True
            import webrtcvad
            self._webrtc_vad = webrtcvad.Vad(aggressiveness)
            self.sr = sample_rate
            return
            
        self._use_webrtc = False
        self.sr = sample_rate
        self.frame_ms = frame_ms
        
        # Map aggressiveness to threshold (0=permissive, 3=strict)
        # Lower threshold = more speech detected
        threshold_map = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        self.threshold = threshold_map.get(aggressiveness, 0.25)
        
        # Load model
        self.model = _load_silero()
        self.model.reset_states()
        
        # Silero needs minimum 512 samples (32ms at 16kHz)
        # We'll accumulate audio in a buffer
        self.min_samples = 512
        self.buffer = np.array([], dtype=np.float32)
    
    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Check if audio frame contains speech.
        
        Args:
            pcm_bytes: Raw PCM audio bytes (int16 little-endian)
            
        Returns:
            True if speech detected, False otherwise
        """
        if self._use_webrtc:
            return self._webrtc_vad.is_speech(pcm_bytes, self.sr)
            
        try:
            # Convert bytes to numpy array (int16 -> float32 normalized)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Add to buffer
            self.buffer = np.concatenate([self.buffer, audio_float])
            
            # Only run inference when we have enough samples
            if len(self.buffer) < self.min_samples:
                return False  # Not enough data yet
            
            # Take exactly min_samples from buffer
            chunk = self.buffer[:self.min_samples]
            self.buffer = self.buffer[self.min_samples:]  # Remove processed samples
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk)
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sr).item()
            
            return speech_prob >= self.threshold
            
        except Exception as e:
            # Fallback: assume no speech on error
            return False
    
    def reset(self):
        """Reset model states and buffer (call between utterances)."""
        if not self._use_webrtc:
            self.model.reset_states()
            self.buffer = np.array([], dtype=np.float32)
