# src/asr/__init__.py
"""
Factory pentru ASR (Speech-to-Text).
Suportă mod local (Whisper) sau remote (HTTP server).
"""
from typing import Optional
from src.core.logger import setup_logger
from .interface import ASRInterface, LocalASR, RemoteASR


def make_asr(cfg_asr: dict, logger=None) -> ASRInterface:
    """
    Creează o instanță ASR bazată pe configurație.
    
    Args:
        cfg_asr: Configurație din asr.yaml
        logger: Logger opțional
        
    Returns:
        ASRInterface: Implementare locală sau remote
    """
    if logger is None:
        logger = setup_logger("asr")
    
    # Verifică mod: local sau remote
    mode = (cfg_asr.get("mode") or "local").lower()
    
    if mode == "remote":
        # Client HTTP către server
        host = cfg_asr.get("remote_host", "localhost")
        port = int(cfg_asr.get("remote_port", 8001))
        timeout = float(cfg_asr.get("remote_timeout", 30.0))
        logger.info(f"🌐 ASR mode=remote, server={host}:{port}")
        return RemoteASR(host=host, port=port, timeout=timeout, logger=logger)
    
    # Mod local - creează engine și îl învelește în LocalASR
    provider = (cfg_asr.get("provider") or "faster").lower()
    
    if provider == "faster":
        from .engine_faster import ASREngine
        engine = ASREngine(
            model_size=cfg_asr.get("model_size", "base"),
            compute_type=cfg_asr.get("compute_type", "int8"),
            device=cfg_asr.get("device", "cpu"),
            force_language=cfg_asr.get("force_language"),
            beam_size=int(cfg_asr.get("beam_size", 1)),
            vad_min_silence_ms=int(cfg_asr.get("vad_min_silence_ms", 300)),
            warmup_enabled=bool(cfg_asr.get("warmup_enabled", True)),
            logger=logger,
        )
        return LocalASR(engine)
        
    else:
        raise ValueError(f"Unknown ASR provider: {provider}")
