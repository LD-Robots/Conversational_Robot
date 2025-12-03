from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - import guard
    ort = None  # type: ignore[assignment]
    _onnx_error = exc  # type: ignore[var-annotated]
else:
    _onnx_error = None  # type: ignore[var-annotated]

try:
    import torch
    import torchaudio
except Exception as exc:  # pragma: no cover - import guard
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
    _torch_error = exc  # type: ignore[var-annotated]
else:
    _torch_error = None  # type: ignore[var-annotated]


@dataclass
class StopDetectionResult:
    probability: float
    logits: Tuple[float, float]


class StopKeywordDetector:
    """
    Rulează modelul ONNX „stop” și emite un event când scorul depășește pragurile.
    Fereastră: 1s, hop: 0.5s (configurabil).
    """

    def __init__(self, cfg: dict, sample_rate: int, logger):
        if ort is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"onnxruntime indisponibil: {_onnx_error}")  # type: ignore[arg-type]
        if torch is None or torchaudio is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"torch/torchaudio lipsesc: {_torch_error}")  # type: ignore[arg-type]

        self.log = logger
        self.sample_rate = sample_rate
        self.frame = int(cfg.get("frame_samples") or sample_rate)
        self.hop = int(cfg.get("hop_samples") or max(1, self.frame // 2))
        self.logit_margin = float(cfg.get("logit_margin", 0.5))
        self.prob_threshold = float(cfg.get("prob_threshold", 0.8))
        self.hits_required = max(1, int(cfg.get("hits_required", 1)))
        self.debug = bool(cfg.get("debug", False))
        model_path = Path(cfg.get("model_path") or "voices/stop_keyword.onnx").expanduser()

        if self.sample_rate != 16000:
            raise ValueError("StopKeywordDetector necesită sample_rate = 16000 Hz pentru acest model.")
        if not model_path.exists():
            raise FileNotFoundError(f"StopKeywordDetector: model absent ({model_path}).")

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(cfg.get("num_threads", 1)))
        self.session = ort.InferenceSession(str(model_path), so)
        self.input_name = self.session.get_inputs()[0].name
        self._mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=40)
        self._db = torchaudio.transforms.AmplitudeToDB()

        self._buf = np.zeros(self.frame, dtype=np.float32)
        self._buf_filled = False
        self._filled = 0
        self._stride = 0
        self._consecutive_hits = 0

        self.log.info(
            f"🛑 StopKeywordDetector activ: model={model_path.name}, "
            f"frame={self.frame} samp, hop={self.hop} samp, hits_required={self.hits_required}"
        )

    def reset(self) -> None:
        self._stride = 0
        self._consecutive_hits = 0
        self._buf[:] = 0.0
        self._buf_filled = False
        self._filled = 0

    def process_block(self, pcm_i16: np.ndarray) -> Optional[StopDetectionResult]:
        """
        Primește un bloc PCM (int16) și rulează detectorul la fiecare hop.
        Returnează StopDetectionResult doar când pragurile sunt atinse.
        """
        if pcm_i16.size == 0:
            return None

        chunk = pcm_i16.astype(np.float32) / 32768.0
        need = len(chunk)
        if need >= self.frame:
            self._buf[:] = chunk[-self.frame :]
            self._filled = self.frame
            self._buf_filled = True
        else:
            self._buf = np.roll(self._buf, -need)
            self._buf[-need:] = chunk
            self._filled = min(self.frame, self._filled + need)
            self._buf_filled = self._filled >= self.frame

        self._stride += len(chunk)
        result: Optional[StopDetectionResult] = None
        while self._stride >= self.hop:
            self._stride -= self.hop
            if not self._buf_filled:
                continue
            result = self._run_detector(self._buf)
            if result:
                break
        return result

    def _run_detector(self, chunk: np.ndarray) -> Optional[StopDetectionResult]:
        feats = self._featurize(chunk)
        logits = self.session.run(None, {self.input_name: feats})[0][0]
        other = float(logits[0])
        stop = float(logits[1])
        p_other, p_stop = self._softmax2(other, stop)
        raw_hit = (stop - other) >= self.logit_margin and p_stop >= self.prob_threshold
        if raw_hit:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if self.debug:
            self.log.debug(
                f"[STOP-KWS] logits other={other:.3f} stop={stop:.3f} "
                f"p_stop={p_stop:.3f} hits={self._consecutive_hits}/{self.hits_required}"
            )

        if self._consecutive_hits >= self.hits_required:
            self._consecutive_hits = 0
            return StopDetectionResult(probability=p_stop, logits=(other, stop))
        return None

    def _featurize(self, chunk: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(chunk[np.newaxis, :])
        mel = self._mel(t)
        mel_db = self._db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = mel_db.unsqueeze(0)
        feats = mel_db.numpy().astype(np.float32)
        return feats

    @staticmethod
    def _softmax2(a: float, b: float) -> Tuple[float, float]:
        m = max(a, b)
        ea = math.exp(a - m)
        eb = math.exp(b - m)
        s = ea + eb
        if s == 0.0:
            return 0.5, 0.5
        return ea / s, eb / s
