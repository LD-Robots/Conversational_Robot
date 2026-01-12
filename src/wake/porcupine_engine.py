# src/wake/porcupine_engine.py
"""
Porcupine wake word engine - wrapper pentru pvporcupine.
Permite detectarea wake words din fișiere .ppn antrenate cu Picovoice.
"""
from __future__ import annotations

import os
import queue
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import sounddevice as sd

from src.audio.devices import choose_input_device


class PorcupineEngine:
    """
    Wake word engine folosind Porcupine (Picovoice).
    Suportă fișiere .ppn custom antrenate pe console.picovoice.ai
    """
    
    def __init__(self, cfg_audio: Dict[str, Any], cfg_porcupine: Dict[str, Any], logger=None):
        self.cfg_audio = cfg_audio or {}
        self.cfg = cfg_porcupine or {}
        self.log = logger
        
        self._sample_rate = 16000  # Porcupine requires 16kHz
        self._frame_length = 512   # Porcupine frame size
        
        # Get access key
        self._access_key = (
            os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
            or self.cfg.get("access_key", "").strip()
        )
        if not self._access_key:
            raise ValueError("Porcupine: lipsește access_key. Setează PICOVOICE_ACCESS_KEY sau pune în wake.yaml.")
        
        # Parse keywords from config
        self._keywords: Dict[str, Dict[str, Any]] = {}
        self._parse_keywords()
        
        if not self._keywords:
            raise ValueError("Porcupine: nu sunt definite keywords în config (porcupine.keywords).")
        
        # Initialize Porcupine
        self._porcupine = None
        self._keyword_names = []
        self._init_porcupine()
        
        # Audio queue and stream
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._open_stream()
    
    def _parse_keywords(self):
        """Parse keywords din config."""
        kw_cfg = self.cfg.get("keywords") or {}
        for name, opts in kw_cfg.items():
            if isinstance(opts, dict):
                model_path = opts.get("model_path", "")
                sensitivity = float(opts.get("sensitivity", 0.5))
            else:
                model_path = str(opts)
                sensitivity = 0.5
            
            # Resolve path
            if model_path:
                p = Path(model_path).expanduser()
                if not p.is_absolute():
                    p = Path.cwd() / p
                if not p.exists():
                    if self.log:
                        self.log.warning(f"Porcupine: model lipsă pentru '{name}': {p}")
                    continue
                self._keywords[name] = {
                    "model_path": str(p),
                    "sensitivity": sensitivity,
                    "cooldown_ms": int((opts if isinstance(opts, dict) else {}).get("cooldown_ms", 2000)),
                    "last_trigger": 0.0,
                }
    
    def _init_porcupine(self):
        """Inițializează engine-ul Porcupine."""
        try:
            import pvporcupine
        except ImportError as exc:
            raise RuntimeError("Biblioteca `pvporcupine` lipsește. Rulează `pip install pvporcupine`.") from exc
        
        keyword_paths = []
        sensitivities = []
        self._keyword_names = []
        
        for name, kw in self._keywords.items():
            keyword_paths.append(kw["model_path"])
            sensitivities.append(kw["sensitivity"])
            self._keyword_names.append(name)
        
        try:
            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                keyword_paths=keyword_paths,
                sensitivities=sensitivities,
            )
            self._frame_length = self._porcupine.frame_length
            self._sample_rate = self._porcupine.sample_rate
            
            if self.log:
                self.log.info(f"🐍 Porcupine init: keywords={self._keyword_names}, frame={self._frame_length}")
        except Exception as exc:
            raise RuntimeError(f"Porcupine: eroare la inițializare ({exc})") from exc
    
    def _open_stream(self):
        """Deschide stream-ul audio."""
        device = choose_input_device(
            hint=self.cfg_audio.get("input_device_hint", ""),
            logger=self.log
        )
        
        def _callback(indata, frames, time_info, status):
            if status and self.log:
                self.log.debug(f"Porcupine audio status: {status}")
            # Convert to int16
            audio_int16 = (indata[:, 0] * 32768).astype(np.int16)
            self._audio_queue.put(audio_int16)
        
        try:
            self._stream = sd.InputStream(
                device=device,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._frame_length,
                callback=_callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"Porcupine: nu pot deschide microfonul ({exc})") from exc
    
    def available_keywords(self):
        """Returnează lista de keywords disponibile."""
        return list(self._keywords.keys())
    
    def has_keyword(self, keyword_id: str) -> bool:
        """Verifică dacă keyword-ul există."""
        return keyword_id in self._keywords
    
    def wait_for(self, keyword_id: str, timeout_seconds: Optional[float] = None) -> bool:
        """
        Așteaptă detecția unui keyword specific.
        Returnează True dacă s-a detectat, False la timeout.
        """
        if keyword_id not in self._keywords:
            raise ValueError(f"Keyword necunoscut pentru Porcupine: {keyword_id}")
        
        kw_cfg = self._keywords[keyword_id]
        kw_index = self._keyword_names.index(keyword_id)
        
        start = time.time()
        audio_buffer = np.array([], dtype=np.int16)
        
        while True:
            if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
                return False
            
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Process when we have enough samples
                while len(audio_buffer) >= self._frame_length:
                    frame = audio_buffer[:self._frame_length]
                    audio_buffer = audio_buffer[self._frame_length:]
                    
                    result = self._porcupine.process(frame)
                    
                    if result >= 0:
                        detected_name = self._keyword_names[result]
                        if detected_name == keyword_id:
                            # Check cooldown
                            now = time.time()
                            last = kw_cfg.get("last_trigger", 0.0)
                            cooldown_s = kw_cfg.get("cooldown_ms", 2000) / 1000.0
                            
                            if (now - last) >= cooldown_s:
                                kw_cfg["last_trigger"] = now
                                if self.log:
                                    self.log.info(f"🐍 Wake (porcupine:{keyword_id})")
                                return True
                            
            except queue.Empty:
                continue
            except Exception as exc:
                if self.log:
                    self.log.error(f"Porcupine: eroare la procesare ({exc})")
                return False
    
    def wait_for_any(self, timeout_seconds: Optional[float] = None) -> Optional[str]:
        """
        Așteaptă detecția oricărui keyword.
        Returnează numele keyword-ului detectat sau None la timeout.
        """
        start = time.time()
        audio_buffer = np.array([], dtype=np.int16)
        
        while True:
            if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
                return None
            
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                while len(audio_buffer) >= self._frame_length:
                    frame = audio_buffer[:self._frame_length]
                    audio_buffer = audio_buffer[self._frame_length:]
                    
                    result = self._porcupine.process(frame)
                    
                    if result >= 0:
                        detected_name = self._keyword_names[result]
                        kw_cfg = self._keywords[detected_name]
                        
                        # Check cooldown
                        now = time.time()
                        last = kw_cfg.get("last_trigger", 0.0)
                        cooldown_s = kw_cfg.get("cooldown_ms", 2000) / 1000.0
                        
                        if (now - last) >= cooldown_s:
                            kw_cfg["last_trigger"] = now
                            if self.log:
                                self.log.info(f"🐍 Wake (porcupine:{detected_name})")
                            return detected_name
                            
            except queue.Empty:
                continue
            except Exception as exc:
                if self.log:
                    self.log.error(f"Porcupine: eroare la procesare ({exc})")
                return None
    
    def close(self):
        """Eliberează resursele."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
