# src/audio/input.py
import queue, time, struct
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

from .devices import choose_input_device, list_input_devices
from .vad import VAD
from .processing import AudioEffects

# Import opțional: nu crăpa dacă nu există webrtc AEC
try:
    from .aec_webrtc import WebRTCAEC  # opțional
except Exception:
    WebRTCAEC = None


def _float_to_int16(audio_f32: np.ndarray) -> np.ndarray:
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    return (audio_f32 * 32767.0).astype(np.int16)


def record_until_silence(cfg_audio: dict, out_wav_path: Path, logger, quiet_short: bool = False):
    """
    Înregistrează mono 16kHz și se oprește după `silence_ms_to_end` ms de liniște
    (detectată de VAD) sau după `max_record_seconds` (fallback).

    Anti-spam: dacă vocea cumulată < `min_valid_seconds` -> NU salvează fișierul, întoarce voice_sec.
    
    Args:
        quiet_short: Dacă True, nu loghează "utterance prea scurt" (pentru grupare externă)

    Returnează: (path, voice_seconds)
    """
    sr = int(cfg_audio["sample_rate"])
    block_ms = int(cfg_audio["block_ms"])              # 10/20/30 ms
    silence_ms_to_end = int(cfg_audio["silence_ms_to_end"])
    max_secs = int(cfg_audio["max_record_seconds"])
    min_valid_seconds = float(cfg_audio.get("min_valid_seconds", 0.5))

    assert block_ms in (10, 20, 30), "VAD frame must be 10/20/30 ms"
    block_size = int(sr * (block_ms / 1000.0))

    # ——— Audio Effects simple ———
    effects = AudioEffects(
        ns=bool(cfg_audio.get("ns", True)),
        agc=bool(cfg_audio.get("agc", True)),
        hpf=bool(cfg_audio.get("hpf", True)),
    )

    # ——— AEC WebRTC (opțional, doar dacă aec_mode=webrtc și clasa există) ———
    aec = None
    if (str(cfg_audio.get("aec_mode", "system")).lower() == "webrtc") and WebRTCAEC:
        try:
            aec = WebRTCAEC(sample_rate=sr, frame_ms=block_ms)
            logger.info("🔁 WebRTC AEC activ (in-app).")
        except Exception as e:
            logger.warning(f"Nu pot porni WebRTC AEC: {e}. Continui fără AEC.")
            aec = None
    else:
        # Folosești AEC de sistem (PulseAudio/pipewire echo-cancel) dacă e disponibil
        pass

    # ——— Device selection ———
    dev_index = choose_input_device(
        prefer_echo_cancel=bool(cfg_audio.get("prefer_echo_cancel", True)),
        hint=str(cfg_audio.get("input_device_hint", "") or ""),
        index=(cfg_audio.get("input_device_index") if cfg_audio.get("input_device_index") not in (None, "") else None),
        logger=logger
    )

    q = queue.Queue()
    vad = VAD(sr, cfg_audio.get("vad_aggressiveness", 2), block_ms)

    logger.info(f"🎤 Vorbește… (se oprește după {silence_ms_to_end}ms de liniște)")
    started = time.time()
    last_voice_ms = 0
    voiced_ms_total = 0       # — cumulăm DOAR timpul de voce detectată (anti-spam)
    collected = []

    def callback(indata, frames, time_info, status):
        if status:
            logger.debug(f"Audio status: {status}")
        q.put(indata.copy())

    with sd.InputStream(
        channels=1,
        samplerate=sr,
        blocksize=block_size,
        dtype="float32",
        callback=callback,
        device=dev_index  # <- poate fi None (default OS)
    ):
        while True:
            try:
                block = q.get(timeout=0.5)  # float32 [-1,1], mono
            except queue.Empty:
                if time.time() - started > max_secs:
                    break
                continue

            pcm_i16 = _float_to_int16(block[:, 0])

            # AEC (opțional, dacă există)
            if aec:
                try:
                    pcm_i16 = aec.process_frame(pcm_i16)
                except Exception:
                    pass

            # Igienă audio înainte de VAD
            pcm_i16 = effects.process_frame(pcm_i16)

            collected.append(pcm_i16)

            # VAD pe bytes little-endian
            pcm_bytes = struct.pack("<%dh" % len(pcm_i16), *pcm_i16)
            if vad.is_speech(pcm_bytes):
                last_voice_ms = 0
                voiced_ms_total += block_ms
            else:
                last_voice_ms += block_ms

            if last_voice_ms >= silence_ms_to_end:
                break
            if time.time() - started > max_secs:
                break

    if aec:
        try:
            aec.close()
        except Exception:
            pass

    audio = np.concatenate(collected, axis=0) if collected else np.zeros(1, dtype=np.int16)
    voice_sec = voiced_ms_total / 1000.0

    # — dacă vocea efectivă este sub prag -> NU salvăm nimic (anti-spam)
    if voice_sec < min_valid_seconds:
        if not quiet_short:
            logger.info(f"⏭️ Utterance prea scurt (voce ~{voice_sec:.2f}s < {min_valid_seconds:.2f}s) — ignor, nu salvez.")
        return str(out_wav_path), voice_sec

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav_path), audio, sr, subtype="PCM_16")
    dur = len(audio) / sr
    logger.info(f"✅ Înregistrare salvată: {out_wav_path} (audio ~{dur:.2f}s, voce ~{voice_sec:.2f}s)")
    return str(out_wav_path), voice_sec
