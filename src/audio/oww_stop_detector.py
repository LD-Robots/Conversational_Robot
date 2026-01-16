# src/audio/oww_stop_detector.py
"""
OpenWakeWord-based stop keyword detector.
Uses the same embedding/melspec as wake words for ONNX models trained with OpenWakeWord.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np


@dataclass
class OWWStopResult:
    keyword: str
    score: float


class OWWStopDetector:
    """
    Detector pentru cuvinte stop bazat pe OpenWakeWord.
    Suportă multiple modele ONNX (ex: shut_up.onnx, stop_now.onnx).
    """

    def __init__(self, cfg: dict, sample_rate: int, logger):
        self.log = logger
        self.sample_rate = sample_rate
        self.threshold = float(cfg.get("threshold", 0.5))
        self.hits_required = max(1, int(cfg.get("hits_required", 2)))
        self.cooldown_ms = int(cfg.get("cooldown_ms", 1500))
        self.debug = bool(cfg.get("debug", False))

        # Parse models from config
        self._models_cfg = self._parse_models(cfg)
        if not self._models_cfg:
            raise ValueError("OWWStopDetector: no models configured")

        # Init OpenWakeWord
        self._model = self._init_oww()

        # Detection state per keyword
        self._consecutive_hits: Dict[str, int] = {name: 0 for name in self._models_cfg}
        self._last_hit_time: Dict[str, float] = {name: 0.0 for name in self._models_cfg}

        model_names = list(self._models_cfg.keys())
        self.log.info(f"🛑 OWWStopDetector activ: models={model_names}, threshold={self.threshold}, hits={self.hits_required}")

    def _parse_models(self, cfg: dict) -> Dict[str, Dict[str, Any]]:
        """Parse models from config - supports both single model_path and multiple models dict."""
        models: Dict[str, Dict[str, Any]] = {}

        # Check for 'models' dict first
        models_cfg = cfg.get("models") or {}
        if models_cfg:
            for name, mcfg in models_cfg.items():
                path = Path(str(mcfg.get("model_path", ""))).expanduser()
                if path.exists():
                    models[name] = {
                        "path": str(path),
                        "label": mcfg.get("label") or path.stem,
                        "threshold": float(mcfg.get("threshold", self.threshold)),
                    }
                else:
                    self.log.warning(f"OWWStopDetector: model {name} not found at {path}")

        # Fallback to single model_path
        if not models:
            model_path = cfg.get("model_path")
            if model_path:
                path = Path(str(model_path)).expanduser()
                if path.exists():
                    name = path.stem.replace("_", " ").title().replace(" ", "")
                    models[name] = {
                        "path": str(path),
                        "label": path.stem,
                        "threshold": self.threshold,
                    }

        return models

    def _init_oww(self):
        try:
            from openwakeword import Model
        except ImportError as exc:
            raise RuntimeError("openwakeword not installed") from exc

        model_paths = [m["path"] for m in self._models_cfg.values()]

        try:
            return Model(
                wakeword_models=model_paths,
                inference_framework="onnx",
            )
        except FileNotFoundError as exc:
            hint = "Run: python -c \"from openwakeword import utils; utils.download_models()\""
            raise RuntimeError(f"OWWStopDetector: missing resources ({exc}). {hint}") from exc
        except Exception as exc:
            raise RuntimeError(f"OWWStopDetector: init failed ({exc})") from exc

    def reset(self) -> None:
        """Reset detection state."""
        for name in self._consecutive_hits:
            self._consecutive_hits[name] = 0
        try:
            self._model.reset()
        except Exception:
            pass

    def process_block(self, pcm_i16: np.ndarray) -> Optional[OWWStopResult]:
        """
        Process audio block and return OWWStopResult if stop keyword detected.
        """
        if pcm_i16.size == 0:
            return None

        # OpenWakeWord expects int16 samples
        samples = pcm_i16.astype(np.int16).flatten()

        try:
            predictions = self._model.predict(samples)
        except Exception as exc:
            if self.debug:
                self.log.debug(f"OWWStopDetector predict error: {exc}")
            return None

        if not predictions:
            return None

        now = time.monotonic()

        for name, model_cfg in self._models_cfg.items():
            label = model_cfg["label"]
            threshold = model_cfg["threshold"]

            score = predictions.get(label, 0.0)

            if self.debug:
                self.log.debug(f"[OWW-STOP] {name}: score={score:.3f} thr={threshold:.2f} hits={self._consecutive_hits[name]}/{self.hits_required}")

            if score >= threshold:
                self._consecutive_hits[name] += 1
            else:
                self._consecutive_hits[name] = 0

            # Check if we have enough consecutive hits
            if self._consecutive_hits[name] >= self.hits_required:
                # Check cooldown
                last_hit = self._last_hit_time.get(name, 0.0)
                if (now - last_hit) * 1000 >= self.cooldown_ms:
                    self._consecutive_hits[name] = 0
                    self._last_hit_time[name] = now
                    self._model.reset()
                    return OWWStopResult(keyword=name, score=score)

        return None
