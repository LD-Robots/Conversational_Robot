# src/asr/__init__.py
from typing import Optional
from src.core.logger import setup_logger

def make_asr(cfg_asr: dict, logger=None):
    if logger is None:
        logger = setup_logger("asr")
    provider = (cfg_asr.get("provider") or "faster").lower()
    if provider == "faster":
        from .engine_faster import ASREngine
        return ASREngine(
            model_size=cfg_asr.get("model_size", "base"),
            compute_type=cfg_asr.get("compute_type", "int8"),
            device=cfg_asr.get("device", "cpu"),
            force_language=cfg_asr.get("force_language"),
            beam_size=int(cfg_asr.get("beam_size", 1)),
            vad_min_silence_ms=int(cfg_asr.get("vad_min_silence_ms", 300)),
            warmup_enabled=bool(cfg_asr.get("warmup_enabled", True)),
            logger=logger,
        )
    elif provider == "openai":
        from .engine_openai import ASREngine
        return ASREngine(
            model_size=cfg_asr.get("model_size", "base"),
            compute_type=cfg_asr.get("compute_type"),
            device=cfg_asr.get("device", "cpu"),
            force_language=cfg_asr.get("force_language"),
        )
    else:
        raise ValueError(f"Unknown ASR provider: {provider}")
