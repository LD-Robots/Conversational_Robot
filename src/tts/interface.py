# src/tts/interface.py
"""
Interfețe abstracte pentru TTS (Text-to-Speech).
Permite rularea fie local (Edge TTS, Piper), fie remote (HTTP server).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Callable


class TTSInterface(ABC):
    """Interfață abstractă pentru Text-to-Speech."""
    
    @abstractmethod
    def is_speaking(self) -> bool:
        """Returnează True dacă TTS-ul vorbește activ."""
        pass
    
    @abstractmethod
    def say(self, text: str, lang: str = "en"):
        """
        Spune text blocking (sincron).
        
        Args:
            text: Textul de pronunțat
            lang: Limba (en/ro)
        """
        pass
    
    @abstractmethod
    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        """
        Streaming async: consumă tokens de la LLM și le vorbește pe măsură.
        
        Args:
            token_iter: Generator de tokens (string-uri) de la LLM
            lang: Limba pentru voce
            on_first_speak: Callback apelat când începe primul playback
            min_chunk_chars: Număr minim de caractere pe chunk
            on_done: Callback apelat când termină tot
        """
        pass
    
    @abstractmethod
    def say_cached(self, key: str, lang: str = "en") -> bool:
        """
        Redă un răspuns pre-cache-uit.
        
        Args:
            key: Cheia din cache
            lang: Limba
            
        Returns:
            True dacă a găsit și redat, False altfel
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Oprește TTS-ul imediat."""
        pass


class LocalTTS(TTSInterface):
    """
    Implementare locală folosind TTSLocal/EdgeTTS existent.
    Wrapper peste engine-ul existent.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: Instanță de TTSLocal sau EdgeTTS
        """
        self._engine = engine
    
    def is_speaking(self) -> bool:
        return self._engine.is_speaking()
    
    def say(self, text: str, lang: str = "en"):
        self._engine.say(text, lang)
    
    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        self._engine.say_async_stream(
            token_iter, lang, on_first_speak, min_chunk_chars, on_done
        )
    
    def say_cached(self, key: str, lang: str = "en") -> bool:
        return self._engine.say_cached(key, lang)
    
    def stop(self):
        self._engine.stop()


class RemoteTTS(TTSInterface):
    """
    Implementare remote - primește audio generat de server și îl redă local.
    Serverul face sinteza, clientul face doar playback.
    """
    
    def __init__(self, host: str, port: int, timeout: float = 30.0, logger=None):
        """
        Args:
            host: Adresa IP sau hostname a serverului
            port: Portul serverului
            timeout: Timeout pentru request
            logger: Logger opțional
        """
        import threading
        import tempfile
        
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.log = logger
        self._speaking = False
        self._stop_flag = threading.Event()
        self._temp_dir = tempfile.mkdtemp(prefix="remote_tts_")
    
    def is_speaking(self) -> bool:
        return self._speaking
    
    def _play_audio_file(self, path: str):
        """Redă un fișier audio folosind ffplay."""
        import subprocess
        import time
        
        if self._stop_flag.is_set():
            return
        
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            while proc.poll() is None:
                if self._stop_flag.is_set():
                    proc.terminate()
                    proc.wait(timeout=1)
                    return
                time.sleep(0.1)
                
        except Exception as e:
            if self.log:
                self.log.error(f"RemoteTTS playback error: {e}")
    
    def say(self, text: str, lang: str = "en"):
        import requests
        import os
        
        if not text.strip():
            return
        
        self._speaking = True
        self._stop_flag.clear()
        
        try:
            response = requests.post(
                f"{self.base_url}/synthesize",
                json={"text": text, "lang": lang},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Salvează audio-ul primit
            audio_path = os.path.join(self._temp_dir, "response.mp3")
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            
            self._play_audio_file(audio_path)
            
        except requests.exceptions.RequestException as e:
            if self.log:
                self.log.error(f"RemoteTTS error: {e}")
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
        Pentru streaming remote, consumăm toate token-urile și trimitem textul complet.
        Serverul generează audio și noi îl redăm.
        """
        import threading
        
        def worker():
            self._speaking = True
            self._stop_flag.clear()
            
            try:
                # Consumă toate token-urile
                full_text = "".join(token_iter)
                
                if not full_text.strip():
                    return
                
                if on_first_speak:
                    on_first_speak()
                
                # Trimite la server
                self.say(full_text, lang)
                
            finally:
                self._speaking = False
                if on_done:
                    try:
                        on_done()
                    except Exception:
                        pass
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def say_cached(self, key: str, lang: str = "en") -> bool:
        # Remote TTS nu suportă cache local - serverul ar trebui să-l gestioneze
        return False
    
    def stop(self):
        self._stop_flag.set()
        self._speaking = False
