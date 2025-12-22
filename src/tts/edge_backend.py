# src/tts/edge_backend.py
"""
Edge TTS backend - Microsoft Neural Voices gratuit.
Latență mai mică decât Piper, voci foarte naturale.
"""
from __future__ import annotations
from typing import Dict, Optional, Iterable, Callable
import threading
import tempfile
import asyncio
import os
import re
import time
import queue

import edge_tts
import soundfile as sf
import sounddevice as sd

_SENT_SPLIT = re.compile(r'([.!?…:;]+)\s+')


class EdgeTTS:
    """
    Edge TTS backend cu streaming și dublu-buffer.
    Folosește Microsoft Neural Voices gratuit.
    """
    
    def __init__(self, cfg: Dict, logger):
        self.log = logger
        self.cfg = cfg
        
        # Voci pentru fiecare limbă
        self.voice_en = cfg.get("edge_voice_en", "en-GB-SoniaNeural")
        self.voice_ro = cfg.get("edge_voice_ro", "ro-RO-EmilNeural")
        
        # Rate și pitch
        self.rate = cfg.get("edge_rate", "+0%")
        self.pitch = cfg.get("edge_pitch", "+0Hz")
        
        # Control
        self._speaking = False
        self._stop_flag = threading.Event()
        self._playback_stream: Optional[sd.OutputStream] = None
        self._coord_thread: Optional[threading.Thread] = None
        
        # Cache pentru fraze comune
        self._cache_dir = tempfile.mkdtemp(prefix="edge_cache_")
        self._cache: Dict[str, str] = {}
        
        # Pre-cache frazele comune
        self._precache()
        
        self.log.info(f"Edge TTS: EN={self.voice_en}, RO={self.voice_ro}")
    
    def _pick_voice(self, lang: str) -> str:
        """Alege vocea în funcție de limbă."""
        if lang.lower().startswith("ro"):
            return self.voice_ro
        return self.voice_en
    
    def _precache(self):
        """Pre-generează WAV-uri pentru frazele comune."""
        phrases = self.cfg.get("cache_phrases", {})
        if not phrases:
            return
        
        async def gen_all():
            for key, data in phrases.items():
                text = data.get("text", "")
                lang = data.get("lang", "en")
                if not text:
                    continue
                try:
                    voice = self._pick_voice(lang)
                    out_path = os.path.join(self._cache_dir, f"{key}.mp3")
                    communicate = edge_tts.Communicate(text, voice, rate=self.rate, pitch=self.pitch)
                    await communicate.save(out_path)
                    self._cache[key] = out_path
                except Exception as e:
                    if self.log:
                        self.log.warning(f"Edge TTS cache '{key}' failed: {e}")
        
        try:
            asyncio.run(gen_all())
            self.log.info(f"📦 Edge TTS cache: {len(self._cache)} fraze")
        except Exception as e:
            self.log.warning(f"Edge TTS precache error: {e}")
    
    def is_speaking(self) -> bool:
        return self._speaking
    
    def say_cached(self, key: str, lang: str = "en") -> bool:
        """Redă din cache dacă există."""
        path = self._cache.get(key)
        if path and os.path.exists(path):
            self.log.info(f"🔊 Edge TTS cache play: {key}")
            self._play_audio_file(path)
            return True
        return False
    
    def _play_audio_file(self, path: str):
        """Redă un fișier audio (MP3/WAV) folosind ffplay pentru a evita conflicte cu Vosk."""
        import subprocess
        
        if self._stop_flag.is_set():
            return
        
        try:
            # Folosim ffplay care e independent de sounddevice
            # -nodisp = fără fereastră video
            # -autoexit = ieși când termină
            # -loglevel quiet = fără output
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # Așteaptă să termine, dar verifică stop_flag periodic
            while proc.poll() is None:
                if self._stop_flag.is_set():
                    proc.terminate()
                    proc.wait(timeout=1)
                    return
                import time
                time.sleep(0.1)
                
        except FileNotFoundError:
            # Fallback la sounddevice dacă ffplay nu e disponibil
            try:
                data, sr = sf.read(path, dtype="float32")
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                sd.play(data, sr)
                sd.wait()
            except Exception as e:
                self.log.error(f"Edge TTS playback error: {e}")
        except Exception as e:
            self.log.error(f"Edge TTS ffplay error: {e}")
    
    async def _synth_async(self, text: str, voice: str) -> str:
        """Sintetizează text și returnează calea către fișierul audio."""
        fd, out_path = tempfile.mkstemp(suffix=".mp3", prefix="edge_")
        os.close(fd)
        
        communicate = edge_tts.Communicate(text, voice, rate=self.rate, pitch=self.pitch)
        await communicate.save(out_path)
        return out_path
    
    def _synth_blocking(self, text: str, lang: str) -> Optional[str]:
        """Sinteză blocking."""
        voice = self._pick_voice(lang)
        try:
            return asyncio.run(self._synth_async(text, voice))
        except Exception as e:
            self.log.error(f"Edge TTS synth error: {e}")
            return None
    
    def say(self, text: str, lang: str = "en"):
        """Spune text blocking."""
        if not text.strip():
            return
        
        self._speaking = True
        try:
            path = self._synth_blocking(text, lang)
            if path:
                self._play_audio_file(path)
                try:
                    os.remove(path)
                except Exception:
                    pass
        finally:
            self._speaking = False
    
    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        """
        Streaming async: consumă tokens de la LLM, sintetizează în paralel,
        și redă pe măsură ce are chunk-uri gata.
        """
        self._stop_flag.clear()
        self._speaking = True
        
        synth_queue: queue.Queue = queue.Queue(maxsize=3)
        voice = self._pick_voice(lang)
        
        def producer():
            """Acumulează tokens în propoziții și le sintetizează."""
            buffer = ""
            for tok in token_iter:
                if self._stop_flag.is_set():
                    break
                buffer += tok
                
                # Verifică dacă avem propoziții complete
                parts = _SENT_SPLIT.split(buffer)
                while len(parts) >= 3:
                    sentence = parts[0] + parts[1]
                    parts = parts[2:]
                    
                    if len(sentence.strip()) >= min_chunk_chars:
                        if self.log:
                            self.log.info(f"🧠 LLM→TTS chunk [{len(sentence)}c]: {sentence[:60]}...")
                        try:
                            path = asyncio.run(self._synth_async(sentence.strip(), voice))
                            synth_queue.put(path)
                        except Exception as e:
                            self.log.error(f"Edge synth error: {e}")
                
                buffer = "".join(parts)
            
            # Ultimul chunk
            if buffer.strip() and not self._stop_flag.is_set():
                if self.log:
                    self.log.info(f"🧠 LLM→TTS chunk [{len(buffer)}c]: {buffer.strip()}")
                try:
                    path = asyncio.run(self._synth_async(buffer.strip(), voice))
                    synth_queue.put(path)
                except Exception as e:
                    self.log.error(f"Edge synth error: {e}")
            
            synth_queue.put(None)  # Sentinel
        
        def consumer():
            """Redă audio-urile generate."""
            first_played = False
            
            while True:
                if self._stop_flag.is_set():
                    break
                
                try:
                    path = synth_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if path is None:
                    break
                
                if self._stop_flag.is_set():
                    break
                
                if not first_played:
                    first_played = True
                    self.log.info("🔊 TTS play start (chunk 1)")
                    if on_first_speak:
                        try:
                            on_first_speak()
                        except Exception:
                            pass
                
                self._play_audio_file(path)
                
                try:
                    os.remove(path)
                except Exception:
                    pass
            
            self._speaking = False
            if on_done:
                try:
                    on_done()
                except Exception:
                    pass
        
        def coordinator():
            prod_thread = threading.Thread(target=producer, daemon=True)
            cons_thread = threading.Thread(target=consumer, daemon=True)
            
            prod_thread.start()
            cons_thread.start()
            
            prod_thread.join()
            cons_thread.join()
        
        self._coord_thread = threading.Thread(target=coordinator, daemon=True)
        self._coord_thread.start()
    
    def stop(self):
        """Oprește TTS-ul și cleanup thread-uri."""
        self._stop_flag.set()
        self._speaking = False
        
        # Oprește playback-ul imediat
        try:
            sd.stop()
        except Exception:
            pass
        
        # Așteaptă coordinator thread să se termine (cu timeout)
        if self._coord_thread and self._coord_thread.is_alive():
            self._coord_thread.join(timeout=0.5)
