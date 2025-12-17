#!/usr/bin/env python3
"""
Calibrare audio: redă un ton de test prin ec_speaker și înregistrează ec_mic
pentru a estima leak-ul și pragul recomandat pentru barge-in.

Exemplu:
    PULSE_SOURCE=ec_mic PULSE_SINK=ec_speaker python tools/calibrate_audio.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import sounddevice as sd
from textwrap import dedent


def rms_db(vec: np.ndarray) -> float:
    rms = np.sqrt(np.mean(vec * vec) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def normalize(vec: np.ndarray) -> np.ndarray | None:
    vc = vec.astype(np.float32)
    vc -= np.mean(vc)
    norm = np.linalg.norm(vc)
    if norm < 1e-8:
        return None
    return vc / norm


def max_ncc(a: np.ndarray, b: np.ndarray) -> float:
    n = 1
    target_len = len(a) + len(b)
    while n < target_len:
        n <<= 1
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    corr = np.fft.irfft(fa * np.conjugate(fb), n)
    corr = np.concatenate((corr[-len(b) + 1 :], corr[: len(a)]))
    return float(np.max(np.abs(corr)) / len(a))


def synth_signal(samples: int, sr: int) -> np.ndarray:
    t = np.linspace(0, samples / sr, samples, endpoint=False)
    tone1 = 0.6 * np.sin(2 * np.pi * 440 * t)
    tone2 = 0.4 * np.sin(2 * np.pi * 880 * t)
    noise = 0.15 * np.random.randn(samples)
    signal = tone1 + tone2 + noise
    envelope = np.linspace(0, 1, samples)
    envelope = np.minimum(envelope, envelope[::-1])
    signal *= envelope
    signal /= np.max(np.abs(signal) + 1e-6)
    return signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Calibrare cameră pentru barge-in.")
    parser.add_argument("--duration", type=float, default=25.0, help="durata tonului (secunde)")
    parser.add_argument("--samplerate", type=int, default=16000, help="sample rate (Hz)")
    parser.add_argument("--output", type=Path, default=None, help="salvează înregistrarea în WAV (opțional)")
    args = parser.parse_args()

    sr = args.samplerate
    samples = int(sr * args.duration)
    if samples <= 0:
        print("Durata trebuie să fie > 0", file=sys.stderr)
        sys.exit(1)

    print(
        dedent(
            f"""
            ───────────────────────────────────────────────
            Calibrare audio (~{args.duration:.0f}s)

            1) Asigură-te că routing-ul este ec_speaker/ec_mic.
            2) Taci pe durata testului (lasă doar tonul).
            3) Monitorizează în pavucontrol dacă e nevoie.
            ───────────────────────────────────────────────
            """
        )
    )

    test_signal = synth_signal(samples, sr)
    print("▶️  Pornesc tonul de test (și înregistrarea)...")
    recording = sd.playrec(test_signal, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    rec = recording.flatten()
    print("✅ Înregistrare terminată.")

    if args.output:
        import soundfile as sf

        sf.write(str(args.output), rec, sr)
        print(f"💾 Salvat înregistrarea în {args.output}")

    leak_rms = rms_db(rec)
    sig_rms = rms_db(test_signal)
    leak_ratio = leak_rms - sig_rms

    fft = np.fft.rfft(rec * np.hanning(len(rec)))
    freqs = np.fft.rfftfreq(len(rec), 1.0 / sr)
    power = np.abs(fft) ** 2
    total_power = np.sum(power) + 1e-12
    low_band = np.sum(power[freqs < 200])
    low_ratio = low_band / total_power
    if low_ratio > 0.6:
        hp_suggest = 240
    elif low_ratio > 0.4:
        hp_suggest = 220
    elif low_ratio > 0.25:
        hp_suggest = 200
    else:
        hp_suggest = 160

    barge_rms = round(leak_rms + 5.0, 1)

    print("\n──────── Rezultate ────────")
    print(f"RMS ton:        {sig_rms:.1f} dBFS")
    print(f"RMS leak mic:   {leak_rms:.1f} dBFS  (Δ {leak_ratio:.1f} dB)")
    print(f"Low-band ratio (<200 Hz): {low_ratio*100:.1f}%")

    print("\n──────── Sugestii (editează configs/audio.yaml) ────────")
    print(f"  barge_min_rms_dbfs: {barge_rms}")
    print(f"  barge_highpass_hz:  {hp_suggest}")

    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
