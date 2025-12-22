# src/llm/__init__.py
"""
Factory pentru LLM (Language Model).
Suportă mod local (Ollama, Groq) sau remote (HTTP server).
"""
from typing import Optional
from src.core.logger import setup_logger
from .interface import LLMInterface, LocalLLM, RemoteLLM


def make_llm(cfg_llm: dict, logger=None) -> LLMInterface:
    """
    Creează o instanță LLM bazată pe configurație.
    
    Args:
        cfg_llm: Configurație din llm.yaml
        logger: Logger opțional
        
    Returns:
        LLMInterface: Implementare locală sau remote
    """
    if logger is None:
        logger = setup_logger("llm")
    
    # Verifică mod: local sau remote
    mode = (cfg_llm.get("mode") or "local").lower()
    
    if mode == "remote":
        # Client HTTP către server
        host = cfg_llm.get("remote_host", "localhost")
        port = int(cfg_llm.get("remote_port", 8001))
        timeout = float(cfg_llm.get("remote_timeout", 60.0))
        logger.info(f"🌐 LLM mode=remote, server={host}:{port}")
        return RemoteLLM(host=host, port=port, timeout=timeout, logger=logger)
    
    # Mod local - creează engine și îl învelește în LocalLLM
    from .engine import LLMLocal as LLMEngine
    engine = LLMEngine(cfg_llm, logger)
    return LocalLLM(engine)
