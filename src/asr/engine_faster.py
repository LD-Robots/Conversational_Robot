# src/asr/engine_faster.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import os, tempfile, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from src.telemetry.metrics import observe_hist, asr_latency

class ASREngine:
    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",     # int8 / int8_float16 / float16 / float32
        device: str = "cpu",            # cpu / cuda
        force_language: Optional[str] = None,
        beam_size: int = 1,
        vad_min_silence_ms: int = 300,
        warmup_enabled: bool = True,
        logger=None,
    ):
        self.force_language = (force_language or "").strip().lower() or None
        self.beam_size = int(beam_size or 1)
        self.vad_min_silence_ms = int(vad_min_silence_ms or 300)
        self.warmup_enabled = warmup_enabled
        self.log = logger
        self._warmed_up = False
        
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None,
        )
        print(f"[ASR] faster-whisper model={model_size} device={device} compute_type={compute_type} "
              f"force_language={self.force_language} vad_min_silence_ms={self.vad_min_silence_ms}")
        
        # Warm-up la boot
        self._ensure_warm()

    def _ensure_warm(self):
        """Încarcă complet modelul prin transcriere dummy."""
        if not self.warmup_enabled or self._warmed_up:
            return
        try:
            if self.log:
                self.log.info("🔥 ASR warm-up start")
            start = time.perf_counter()
            
            # Creează fișier audio scurt (0.5s tăcere)
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                silence = np.zeros(8000, dtype=np.float32)  # 0.5s @ 16kHz
                sf.write(temp_path, silence, 16000)
                
                # Transcriere dummy
                self.model.transcribe(temp_path, language="en", beam_size=1)
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
            elapsed = time.perf_counter() - start
            self._warmed_up = True
            if self.log:
                self.log.info(f"✅ ASR warm-up gata ({elapsed:.2f}s)")
        except Exception as e:
            if self.log:
                self.log.warning(f"ASR warm-up eșuat: {e}")


    # ---- helper intern
    def _run_once(self, wav_path: str | Path, language: Optional[str], use_vad: bool) -> Tuple[str, str, float, float]:
        """
        Returnează: (text, lang_out, lang_prob, score)
        score = medie(avg_logprob pe segmente) + 0.01 * len(text)
        """
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms} if use_vad else None,
            no_speech_threshold=0.6,
            log_prob_threshold=-0.5,
            condition_on_previous_text=False,
        )
        segs: List = list(segments)
        text = "".join(s.text for s in segs).strip()
        # scor simplu și robust
        if segs:
            # uneori s.avg_logprob lipsește; folosim -5.0 default
            vals = [getattr(s, "avg_logprob", -5.0) if getattr(s, "avg_logprob", None) is not None else -5.0 for s in segs]
            avg_lp = sum(vals) / len(vals)
        else:
            avg_lp = -9.0
        score = avg_lp + 0.01 * len(text)
        out_lang = info.language or (language or "en")
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, out_lang, prob, score

    # ---- API standard (păstrat, dar robust la bug-ul cu max() pe colecție vidă)
    def transcribe(self, wav_path: str | Path, language_override: Optional[str] = None) -> Dict[str, Any]:
        lang = (language_override or self.force_language or None)
        with observe_hist(asr_latency):
            try:
                text, out_lang, prob, _ = self._run_once(wav_path, lang, use_vad=True)
            except ValueError as e:
                if "max() iterable argument is empty" in str(e):
                    fallback_lang = lang or "en"
                    text, out_lang, prob, _ = self._run_once(wav_path, fallback_lang, use_vad=False)
                else:
                    raise
        return {"text": text, "lang": out_lang, "language_probability": prob}

    # ---- NOU: transcriere strict EN/RO -> alegem cea mai bună
    def transcribe_ro_en(self, wav_path: str | Path) -> Dict[str, Any]:
        with observe_hist(asr_latency):
            # rulăm EN & RO cu VAD intern; dacă dă eroare, retry fără VAD
            def safe(lang):
                try:
                    return self._run_once(wav_path, lang, use_vad=True)
                except ValueError as e:
                    if "max() iterable argument is empty" in str(e):
                        return self._run_once(wav_path, lang, use_vad=False)
                    raise
            en_text, _, _, en_score = safe("en")
            ro_text, _, _, ro_score = safe("ro")

        if (ro_score > en_score) and ro_text:
            return {"text": ro_text, "lang": "ro", "language_probability": 1.0}
        else:
            return {"text": en_text, "lang": "en", "language_probability": 1.0}
