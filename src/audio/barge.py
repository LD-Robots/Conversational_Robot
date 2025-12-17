# src/audio/barge.py - Barge-in inteligent (doar voce umană)
from __future__ import annotations
import sounddevice as sd
import numpy as np
import queue, time, struct, math
from typing import Optional
from .vad import VAD
from .devices import choose_input_device

def _rms_dbfs(pcm_i16: np.ndarray) -> float:
    """Calculează RMS în dBFS."""
    if pcm_i16.size == 0:
        return -120.0
    xf = pcm_i16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(xf * xf) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

def _highpass_filter(pcm_i16: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """
    Filtru high-pass simplu (first-order IIR) pentru a tăia frecvențele joase.
    Elimină zgomotele de tip bătăi în masă (~50-200 Hz).
    """
    if cutoff_hz <= 0:
        return pcm_i16
    
    # Coeficient pentru filtrul IIR: alpha = RC / (RC + dt)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    
    xf = pcm_i16.astype(np.float32)
    y = np.zeros_like(xf)
    y_prev = 0.0
    x_prev = 0.0
    
    for i in range(len(xf)):
        y[i] = alpha * (y_prev + xf[i] - x_prev)
        y_prev = y[i]
        x_prev = xf[i]
    
    return np.clip(y, -32768, 32767).astype(np.int16)

def _zero_crossing_rate(pcm_i16: np.ndarray) -> float:
    """
    Calculează rata de treceri prin zero (ZCR).
    Vocea umană: ZCR moderat (~0.05-0.3)
    Zgomote impulsive: ZCR foarte mare (>0.4)
    Zgomote joase constante: ZCR foarte mic (<0.02)
    """
    if len(pcm_i16) < 2:
        return 0.0
    signs = np.sign(pcm_i16)
    crossings = np.sum(np.abs(np.diff(signs))) / 2.0
    return crossings / (len(pcm_i16) - 1)


class BargeInListener:
    """
    Listener inteligent pentru barge-in:
    - Detectează DOAR voce umană (VAD + RMS + spectral filtering + ZCR)
    - Ignoră bătăi în masă, ecoul TTS, zgomote impulsive
    - Anti-impuls: voce continuă >= barge_min_voice_ms
    """
    def __init__(self, cfg_audio: dict, logger):
        self.log = logger
        self.sr = int(cfg_audio["sample_rate"])
        self.block_ms = int(cfg_audio["block_ms"])
        self.block = int(self.sr * (self.block_ms / 1000.0))

        # ——— Voice hold pentru continuitate ———
        self.voice_hold_ms = int(cfg_audio.get("barge_voice_hold_ms", 200))

        # ——— Praguri temporale ———
        self.min_voice_ms = int(cfg_audio.get("barge_min_voice_ms", 800))
        self.debounce_ms = int(cfg_audio.get("barge_debounce_ms", 150))
        self.cooldown_ms = int(cfg_audio.get("barge_cooldown_ms", 800))
        self.arm_after_ms = int(cfg_audio.get("barge_arm_after_ms", 400))
        self.voice_drop_ms = int(cfg_audio.get("barge_voice_drop_ms", self.block_ms))
        self._t0_ms = int(time.monotonic() * 1000)
        self._last_trigger_ms = 0

        # ——— Dinamica prag RMS în funcție de "leak" (eco TTS) ———
        self.leak_margin_db = float(cfg_audio.get("barge_leak_margin_db", 3.0))
        self.leak_decay_ms = int(cfg_audio.get("barge_leak_decay_ms", max(1200, self.cooldown_ms)))
        self._leak_baseline_dbfs: Optional[float] = None
        self._last_leak_update_ms: int = 0

        # ——— Praguri spectrale/acustice ———
        self.min_rms_dbfs = float(cfg_audio.get("barge_min_rms_dbfs", -28.0))
        self.highpass_hz = float(cfg_audio.get("barge_highpass_hz", 300.0))
        self.zcr_min = float(cfg_audio.get("barge_zcr_min", 0.05))
        self.zcr_max = float(cfg_audio.get("barge_zcr_max", 0.35))

        # ——— Device & VAD ———
        self.dev_index = choose_input_device(
            prefer_echo_cancel=bool(cfg_audio.get("prefer_echo_cancel", True)),
            hint=str(cfg_audio.get("input_device_hint", "") or ""),
            logger=logger
        )
        vad_aggr = int(cfg_audio.get("vad_aggressiveness", 3))  # folosim VAD strict (3)
        self.vad = VAD(self.sr, vad_aggr, self.block_ms)
        self.q = queue.Queue()
        self._open_stream()
        self._voiced_ms = 0
        self._last_user_voice_ms: int = 0
        self.debug_meter = bool(cfg_audio.get("barge_debug_meter", False))
        self._debug_interval_ms = int(cfg_audio.get("barge_debug_interval_ms", 120))
        self._last_meter_ms: int = 0
        self.stop_detector = None
        stop_kw_cfg = cfg_audio.get("stop_keyword") or {}
        if stop_kw_cfg.get("enabled"):
            try:
                from .stop_keyword_detector import StopKeywordDetector

                self.stop_detector = StopKeywordDetector(stop_kw_cfg, self.sr, logger)
            except Exception as exc:
                self.stop_detector = None
                self.log.warning(f"Stop keyword detector dezactivat: {exc}")

        self.log.info(f"🎯 Barge-in inteligent: min_voice={self.min_voice_ms}ms, "
                      f"rms_thr={self.min_rms_dbfs}dB, hp={self.highpass_hz}Hz, "
                      f"zcr=[{self.zcr_min},{self.zcr_max}], vad=WebRTC")

    def _open_stream(self):
        def cb(indata, frames, time_info, status):
            try:
                self.q.put_nowait(indata.copy())
            except:
                pass
        self.stream = sd.InputStream(
            channels=1, samplerate=self.sr, blocksize=self.block,
            dtype="float32", callback=cb, device=self.dev_index
        )
        self.stream.start()

    def _maybe_decay_leak(self, now_ms: int) -> None:
        if self._leak_baseline_dbfs is None:
            return
        if (now_ms - self._last_leak_update_ms) > self.leak_decay_ms:
            self._leak_baseline_dbfs = None
            self._last_leak_update_ms = now_ms

    def _update_leak_baseline(self, rms_db: float, now_ms: int, fast: bool = False) -> None:
        if not math.isfinite(rms_db):
            return
        if rms_db <= -90.0:
            return
        if self._leak_baseline_dbfs is None:
            self._leak_baseline_dbfs = rms_db
        else:
            if not fast and rms_db > self._leak_baseline_dbfs + self.leak_margin_db * 2:
                rms_db = self._leak_baseline_dbfs + self.leak_margin_db * 2
            alpha = 0.35 if fast else 0.12
            self._leak_baseline_dbfs = (1.0 - alpha) * self._leak_baseline_dbfs + alpha * rms_db
        self._last_leak_update_ms = now_ms

    def _debug_meter(
        self,
        rms_db: float,
        zcr: Optional[float],
        detected: bool,
        leak_db: Optional[float] = None
    ) -> None:
        if not self.debug_meter:
            return
        now_ms = int(time.monotonic() * 1000)
        if (now_ms - self._last_meter_ms) < self._debug_interval_ms:
            return
        self._last_meter_ms = now_ms

        label = f"rms={rms_db:.1f}dB"
        span = max(5.0, abs(self.min_rms_dbfs))
        value = max(0.0, min(1.0, (rms_db - self.min_rms_dbfs) / span))

        bar_len = 20
        filled = min(bar_len, int(round(value * bar_len)))
        bar = "#" * filled + "-" * (bar_len - filled)
        zcr_str = f" zcr={zcr:.2f}" if zcr is not None else ""
        leak_str = f" leak={leak_db:.1f}dB" if leak_db is not None else ""
        status = "Y" if detected else "n"
        self.log.info(f"[BARGE] |{bar}| {label}{zcr_str}{leak_str} det={status}")

    def _is_human_voice(self, pcm_i16: np.ndarray) -> bool:
        """
        Verifică dacă PCM-ul conține voce umană (nu zgomot/eco):
        1. RMS peste prag (vocea e mai tare decât TTS leak)
        2. High-pass filter (elimină bătăi joase)
        3. Zero-crossing rate în interval vocii umane
        4. VAD confirmă speech
        """
        now_ms = int(time.monotonic() * 1000)
        self._maybe_decay_leak(now_ms)

        leak_db = self._leak_baseline_dbfs
        rms = _rms_dbfs(pcm_i16)

        rms_threshold = self.min_rms_dbfs
        if leak_db is not None:
            rms_threshold = max(rms_threshold, leak_db + self.leak_margin_db)

        if rms < rms_threshold:
            self._update_leak_baseline(rms, now_ms, fast=False)
            self._debug_meter(rms, None, False, leak_db)
            return False

        # 2) High-pass filtering (anti-zgomot jos-frecvent)
        pcm_filtered = _highpass_filter(pcm_i16, self.highpass_hz, self.sr)

        # 3) Zero-crossing rate (anti-zgomot impulsiv)
        zcr = _zero_crossing_rate(pcm_filtered)
        if not (self.zcr_min <= zcr <= self.zcr_max):
            self._update_leak_baseline(rms, now_ms, fast=False)
            self._debug_meter(rms, zcr, False, leak_db)
            return False

        # 4) VAD final check (WebRTC)
        pcm_bytes = struct.pack("<%dh" % len(pcm_filtered), *pcm_filtered)
        detected = self.vad.is_speech(pcm_bytes)
        if not detected and (now_ms - self._last_user_voice_ms) <= self.voice_hold_ms:
            detected = True

        if detected:
            self._last_user_voice_ms = now_ms
            self._debug_meter(rms, zcr, True, leak_db)
            return True

        self._update_leak_baseline(rms, now_ms, fast=False)
        self._debug_meter(rms, zcr, False, leak_db)
        return False

    def heard_speech(self, need_ms: int = None) -> bool:
        """
        Returnează True dacă a detectat voce umană continuă >= need_ms.
        Ignoră zgomotele, bătăile, ecoul TTS.
        """
        if need_ms is None:
            need_ms = self.min_voice_ms

        now_ms = int(time.monotonic() * 1000)

        # Arm-delay: ignoră totul la început (anti-scurgeri inițiale)
        if (now_ms - self._t0_ms) < self.arm_after_ms:
            try:
                while True:
                    block = self.q.get_nowait()
                    pcm = np.clip(block[:, 0], -1, 1)
                    pcm_i16 = (pcm * 32767.0).astype(np.int16)
                    self._update_leak_baseline(_rms_dbfs(pcm_i16), int(time.monotonic() * 1000), fast=True)
            except queue.Empty:
                pass
            return False

        # Debounce: evită trigger repetat rapid
        if (now_ms - self._last_trigger_ms) < self.debounce_ms:
            return False

        # Procesează frame-uri până la deadline scurt (20ms)
        deadline = time.time() + 0.02
        while time.time() < deadline:
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                break

            pcm = np.clip(block[:, 0], -1, 1)
            pcm_i16 = (pcm * 32767.0).astype(np.int16)

            if self.stop_detector:
                stop_detection = self.stop_detector.process_block(pcm_i16)
                if stop_detection:
                    now_stop = int(time.monotonic() * 1000)
                    self._last_trigger_ms = now_stop
                    self._last_user_voice_ms = now_stop
                    self._voiced_ms = 0
                    other_logit, stop_logit = stop_detection.logits
                    self.log.info(
                        f"🛑 Stop keyword detectat (p_stop={stop_detection.probability:.2f}, "
                        f"logits other={other_logit:.2f} stop={stop_logit:.2f})"
                    )
                    return True

            # Verifică dacă e voce umană (nu zgomot/eco)
            if self._is_human_voice(pcm_i16):
                self._voiced_ms = min(self._voiced_ms + self.block_ms, need_ms)
            else:
                # Pierde progres gradual (nu reset instant) pentru drop-uri scurte
                self._voiced_ms = max(0, self._voiced_ms - self.voice_drop_ms)

            # Trigger dacă voce continuă >= need_ms
            if self._voiced_ms >= need_ms:
                now2 = int(time.monotonic() * 1000)
                # Cooldown: evită dublu-trigger
                if (now2 - self._last_trigger_ms) >= self.cooldown_ms:
                    self._last_trigger_ms = now2
                    self._voiced_ms = 0
                    self.log.info(f"🎤 Barge-in: voce umană detectată ({need_ms}ms)")
                    return True
                self._voiced_ms = 0
                return False

        return False

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def user_is_speaking(self) -> bool:
        """Indică dacă recent a fost detectată voce umană (folosit de FastExit)."""
        now_ms = int(time.monotonic() * 1000)
        return (now_ms - self._last_user_voice_ms) <= self.voice_hold_ms
