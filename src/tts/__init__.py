# src/tts/__init__.py
"""
Factory pentru TTS (Text-to-Speech).
Suportă mod local (Edge TTS, Piper) sau remote (HTTP server).
"""
from typing import Optional
from src.core.logger import setup_logger
from .interface import TTSInterface, LocalTTS, RemoteTTS


def make_tts(cfg_tts: dict, logger=None) -> TTSInterface:
    """
    Creează o instanță TTS bazată pe configurație.
    
    Args:
        cfg_tts: Configurație din tts.yaml
        logger: Logger opțional
        
    Returns:
        TTSInterface: Implementare locală sau remote
    """
    if logger is None:
        logger = setup_logger("tts")
    
    # Verifică mod: local sau remote
    mode = (cfg_tts.get("mode") or "local").lower()
    
    if mode == "remote":
        # Client HTTP către server
        host = cfg_tts.get("remote_host", "localhost")
        port = int(cfg_tts.get("remote_port", 8001))
        timeout = float(cfg_tts.get("remote_timeout", 30.0))
        logger.info(f"🌐 TTS mode=remote, server={host}:{port}")
        
        tts_client = RemoteTTS(host=host, port=port, timeout=timeout, logger=logger)
        
        # Health check la startup
        try:
            import requests
            health_url = f"http://{host}:{port}/health"
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                logger.info(f"✅ TTS server disponibil ({health_url})")
            else:
                logger.warning(f"⚠️ TTS server răspunde cu status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ TTS server nu răspunde la {host}:{port} — pornește serverul!")
        except Exception as e:
            logger.warning(f"⚠️ Health check TTS eșuat: {e}")
        
        return tts_client
    
    # Mod local - creează engine și îl învelește în LocalTTS
    backend = (cfg_tts.get("backend") or "pyttsx3").lower()
    
    if backend == "edge":
        from .edge_backend import EdgeTTS
        engine = EdgeTTS(cfg_tts, logger)
        return LocalTTS(engine)
    else:
        # Folosește TTSLocal care alege între Piper și pyttsx3
        from .engine import TTSLocal as TTSEngine
        engine = TTSEngine(cfg_tts, logger)
        return LocalTTS(engine)
