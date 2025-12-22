# src/asr/interface.py
"""
Interfețe abstracte pentru ASR (Speech-to-Text).
Permite rularea fie local (Whisper), fie remote (HTTP server).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class ASRInterface(ABC):
    """Interfață abstractă pentru Speech-to-Text."""
    
    @abstractmethod
    def transcribe(self, wav_path: str | Path, language_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcrie un fișier audio.
        
        Args:
            wav_path: Calea către fișierul WAV
            language_override: Forțează o limbă specifică (opțional)
            
        Returns:
            Dict cu: {"text": str, "lang": str, "language_probability": float}
        """
        pass
    
    @abstractmethod
    def transcribe_ro_en(self, wav_path: str | Path) -> Dict[str, Any]:
        """
        Transcrie cu detecție automată RO/EN.
        Rulează transcriere în ambele limbi și alege cea mai bună.
        
        Args:
            wav_path: Calea către fișierul WAV
            
        Returns:
            Dict cu: {"text": str, "lang": str, "language_probability": float}
        """
        pass


class LocalASR(ASRInterface):
    """
    Implementare locală folosind faster-whisper.
    Wrapper peste ASREngine existent.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: Instanță de ASREngine (faster-whisper)
        """
        self._engine = engine
    
    def transcribe(self, wav_path: str | Path, language_override: Optional[str] = None) -> Dict[str, Any]:
        return self._engine.transcribe(wav_path, language_override)
    
    def transcribe_ro_en(self, wav_path: str | Path) -> Dict[str, Any]:
        return self._engine.transcribe_ro_en(wav_path)


class RemoteASR(ASRInterface):
    """
    Implementare remote - trimite audio la un server HTTP.
    Va fi folosit pe client când serverul face procesarea.
    """
    
    def __init__(self, host: str, port: int, timeout: float = 30.0, logger=None):
        """
        Args:
            host: Adresa IP sau hostname a serverului
            port: Portul serverului
            timeout: Timeout pentru request (ASR poate dura mult)
            logger: Logger opțional
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.log = logger
    
    def transcribe(self, wav_path: str | Path, language_override: Optional[str] = None) -> Dict[str, Any]:
        import requests
        
        url = f"{self.base_url}/transcribe"
        
        try:
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            
            params = {}
            if language_override:
                params['language'] = language_override
            
            response = requests.post(
                url,
                data=audio_data,
                params=params,
                headers={'Content-Type': 'audio/wav'},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if self.log:
                self.log.error(f"RemoteASR error: {e}")
            return {"text": "", "lang": "en", "language_probability": 0.0}
    
    def transcribe_ro_en(self, wav_path: str | Path) -> Dict[str, Any]:
        import requests
        
        url = f"{self.base_url}/transcribe_ro_en"
        
        try:
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            
            response = requests.post(
                url,
                data=audio_data,
                headers={'Content-Type': 'audio/wav'},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if self.log:
                self.log.error(f"RemoteASR error: {e}")
            return {"text": "", "lang": "en", "language_probability": 0.0}
