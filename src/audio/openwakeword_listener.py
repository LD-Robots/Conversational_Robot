from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import sounddevice as sd

from src.audio.devices import choose_input_device


class OpenWakeWordListener:
    """
    Listener asincron pentru modele OpenWakeWord. Rulează într-un thread de fundal,
    procesează audio continuu și emite un callback când detectează keyword-ul configurat.
    """

    def __init__(
        self,
        cfg_audio: Dict[str, Any],
        cfg_openwake: Dict[str, Any],
        logger=None,
        on_detect: Optional[Callable[[str, float], None]] = None,
    ):
        self.log = logger
        self.cfg_audio = cfg_audio or {}
        self.cfg_openwake = cfg_openwake or {}
        self.on_detect = on_detect

        self.sample_rate = int(self.cfg_openwake.get("sample_rate") or self.cfg_audio.get("sample_rate", 16000))
        block_ms = int(self.cfg_openwake.get("block_ms") or self.cfg_audio.get("block_ms", 20))
        self.block = max(160, int(self.sample_rate * (block_ms / 1000.0)))
        self.inference_framework = (self.cfg_openwake.get("inference_framework") or "onnx").lower()
        self.enable_speex = bool(self.cfg_openwake.get("speex_noise_suppression", False))
        self.vad_threshold = float(self.cfg_openwake.get("vad_threshold", 0.0))

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=int(self.cfg_openwake.get("queue_max", 8)))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._device_index = None
        self._last_error_ts: Optional[float] = None

        self._keywords = self._parse_keywords()
        if not self._keywords:
            raise ValueError("openwakeword listener: definește cel puțin un keyword (keywords/model_path).")

        self._model = self._init_model()

    def _parse_keywords(self) -> Dict[str, Dict[str, Any]]:
        keywords_cfg = self.cfg_openwake.get("keywords") or {}
        fallback_path = self.cfg_openwake.get("model_path")
        fallback_label = self.cfg_openwake.get("label")

        if not keywords_cfg and fallback_path:
            friendly = fallback_label or Path(fallback_path).stem
            keywords_cfg = {
                friendly: {
                    "model_path": fallback_path,
                    "threshold": self.cfg_openwake.get("threshold"),
                    "cooldown_ms": self.cfg_openwake.get("min_gap_ms"),
                }
            }

        parsed: Dict[str, Dict[str, Any]] = {}
        default_thr = float(self.cfg_openwake.get("threshold", 0.5))
        default_cd = int(self.cfg_openwake.get("min_gap_ms", self.cfg_openwake.get("cooldown_ms", 1200)))

        for name, kw_cfg in keywords_cfg.items():
            path = Path(str(kw_cfg.get("model_path", ""))).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"openwakeword listener: modelul pentru '{name}' lipsește: {path}")

            parsed[name] = {
                "id": name,
                "model_path": str(path),
                "label": kw_cfg.get("model_label") or kw_cfg.get("label") or path.stem,
                "threshold": float(kw_cfg.get("threshold", default_thr)),
                "cooldown_ms": int(kw_cfg.get("cooldown_ms", default_cd)),
                "kind": (kw_cfg.get("kind") or "stop").lower(),
                "last_hit": 0.0,
            }

        return parsed

    def _init_model(self):
        try:
            from openwakeword import Model
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Biblioteca `openwakeword` lipsește. Rulează `pip install openwakeword`.") from exc

        model_paths = [kw["model_path"] for kw in self._keywords.values()]

        preproc_kwargs: Dict[str, Any] = {}
        for key in ("melspec_model_path", "embedding_model_path", "device", "ncpu"):
            val = self.cfg_openwake.get(key)
            if val:
                preproc_kwargs[key] = val
        preproc_kwargs["sr"] = self.sample_rate

        try:
            return Model(
                wakeword_models=model_paths,
                inference_framework=self.inference_framework,
                enable_speex_noise_suppression=self.enable_speex,
                vad_threshold=self.vad_threshold,
                **preproc_kwargs,
            )
        except FileNotFoundError as exc:
            hint = "Ai rulat `python -c \"from openwakeword import utils; utils.download_models()\"`?"
            raise RuntimeError(f"openwakeword listener: resurse lipsă ({exc}). {hint}") from exc
        except Exception as exc:
            raise RuntimeError(f"openwakeword listener: nu pot inițializa modelul ({exc}).") from exc

    def _ensure_stream(self):
        if self._stream:
            return
        self._device_index = choose_input_device(
            prefer_echo_cancel=bool(self.cfg_audio.get("prefer_echo_cancel", True)),
            hint=str(self.cfg_audio.get("input_device_hint", "") or ""),
            logger=self.log,
        )

        def _callback(indata, frames, time_info, status):
            if status and self.log:
                self.log.debug(f"OpenWakeWordListener input status: {status}")
            block = indata.copy()
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(block)
                except queue.Full:
                    pass

        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block,
                dtype="int16",
                callback=_callback,
                device=self._device_index,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"openwakeword listener: nu pot deschide stream-ul audio ({exc}).") from exc

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ensure_stream()
        self._thread = threading.Thread(target=self._run, name="OpenWakeWordListener", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            if self._model:
                self._model.reset()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            samples = self._to_mono(block)
            predictions = self._predict(samples)
            if not predictions:
                continue

            for kw in self._keywords.values():
                score = predictions.get(kw["label"])
                if score is None:
                    continue
                if score >= kw["threshold"] and self._cooldown_passed(kw):
                    kw["last_hit"] = time.monotonic()
                    if self.log:
                        self.log.info(f"🛑 OpenWakeWord listener: '{kw['id']}' detectat (score={score:.2f}).")
                    if callable(self.on_detect):
                        try:
                            self.on_detect(kw["id"], score)
                        except Exception as cb_err:
                            if self.log:
                                self.log.warning(f"openwakeword listener callback error: {cb_err}")

    def _predict(self, samples: np.ndarray) -> Dict[str, float]:
        try:
            return self._model.predict(samples)
        except Exception as exc:
            now = time.monotonic()
            if not self._last_error_ts or (now - self._last_error_ts) > 5.0:
                if self.log:
                    self.log.error(f"openwakeword listener: eroare runtime ({exc})")
                self._last_error_ts = now
            return {}

    @staticmethod
    def _to_mono(block: np.ndarray) -> np.ndarray:
        if block.ndim == 2:
            pcm = block[:, 0]
        else:
            pcm = block
        return np.asarray(pcm, dtype=np.int16).flatten()

    @staticmethod
    def _cooldown_passed(keyword_cfg: Dict[str, Any]) -> bool:
        cooldown = max(0, int(keyword_cfg.get("cooldown_ms", 0))) / 1000.0
        if cooldown <= 0:
            return True
        return (time.monotonic() - float(keyword_cfg.get("last_hit", 0.0))) >= cooldown
