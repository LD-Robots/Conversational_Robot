# src/llm/interface.py
"""
Interfețe abstracte pentru LLM (Language Model).
Permite rularea fie local (Ollama), fie remote (HTTP server).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Iterator


class LLMInterface(ABC):
    """Interfață abstractă pentru Language Model."""
    
    @abstractmethod
    def generate(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None) -> str:
        """
        Generează un răspuns complet (non-streaming).
        
        Args:
            user_text: Textul utilizatorului
            lang_hint: Limba preferată pentru răspuns
            mode: Mod de generare (precise/creative)
            
        Returns:
            Răspunsul generat ca string
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self, 
        user_text: str, 
        lang_hint: str = "en", 
        mode: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> Iterator[str]:
        """
        Generează un răspuns cu streaming (token cu token).
        
        Args:
            user_text: Textul utilizatorului
            lang_hint: Limba preferată
            mode: Mod de generare
            history: Istoricul conversației
            
        Returns:
            Generator de tokens (string-uri)
        """
        pass


class LocalLLM(LLMInterface):
    """
    Implementare locală folosind LLMLocal existent.
    Suportă Ollama, Groq.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: Instanță de LLMLocal
        """
        self._engine = engine
    
    def generate(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None) -> str:
        return self._engine.generate(user_text, lang_hint, mode)
    
    def generate_stream(
        self, 
        user_text: str, 
        lang_hint: str = "en", 
        mode: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> Iterator[str]:
        return self._engine.generate_stream(user_text, lang_hint, mode, history)


class RemoteLLM(LLMInterface):
    """
    Implementare remote - trimite cereri la un server HTTP.
    Va fi folosit pe client când serverul face procesarea.
    """
    
    def __init__(self, host: str, port: int, timeout: float = 60.0, logger=None):
        """
        Args:
            host: Adresa IP sau hostname a serverului
            port: Portul serverului
            timeout: Timeout pentru request
            logger: Logger opțional
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.log = logger
    
    def generate(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None) -> str:
        import requests
        
        url = f"{self.base_url}/generate"
        
        try:
            response = requests.post(
                url,
                json={
                    "text": user_text,
                    "lang": lang_hint,
                    "mode": mode
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
            
        except requests.exceptions.RequestException as e:
            if self.log:
                self.log.error(f"RemoteLLM error: {e}")
            return ""
    
    def generate_stream(
        self, 
        user_text: str, 
        lang_hint: str = "en", 
        mode: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> Iterator[str]:
        import requests
        
        url = f"{self.base_url}/generate_stream"
        
        try:
            response = requests.post(
                url,
                json={
                    "text": user_text,
                    "lang": lang_hint,
                    "mode": mode,
                    "history": history or []
                },
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Streaming: citește linie cu linie
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    yield line
                    
        except requests.exceptions.RequestException as e:
            if self.log:
                self.log.error(f"RemoteLLM stream error: {e}")
            return
