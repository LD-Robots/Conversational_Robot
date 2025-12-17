#!/usr/bin/env python3
"""
Rulează un model OpenWakeWord pe un fișier WAV și afișează scorurile.

Exemplu:
    python tools/test_openwakeword.py \
        --model voices/stop_now.onnx \
        --audio sample_stop_now.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import onnxruntime as ort
from python_speech_features import logfbank
from openwakeword import Model


def load_audio(path: Path, expected_sr: int) -> np.ndarray:
    pcm, sr = sf.read(str(path), dtype="float32")
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1)
    if sr != expected_sr:
        raise ValueError(
            f"WAV trebuie să fie {expected_sr} Hz (ai {sr} Hz). Folosește resampling sau reînregistrează."
        )
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Testează scorurile OpenWakeWord / ONNX KWS pentru un fișier WAV.")
    parser.add_argument("--model", required=True, help="calea către modelul ONNX (de ex. voices/stop_now.onnx)")
    parser.add_argument("--audio", required=True, help="fișier WAV mono 16kHz cu fraza de test")
    parser.add_argument("--framework", default="onnx", help="inference framework (default: onnx)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="sample rate așteptat (default 16k)")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=160,
        help="dimensiunea blocului trimis la model (OpenWakeWord: 160 ≈10ms, PyTorch KWS: setați 16000)",
    )
    parser.add_argument(
        "--stop-model",
        action="store_true",
        help="Folosește un model PyTorch/ONNX binary (other/stop) și calculează scoruri via onnxruntime.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    audio_path = Path(args.audio).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Modelul nu există: {model_path}")
    if not audio_path.exists():
        raise SystemExit(f"Fișierul audio nu există: {audio_path}")

    samples = load_audio(audio_path, args.sample_rate)

    if args.stop_model:
        session = ort.InferenceSession(str(model_path))
        frame = max(args.frame_size, args.sample_rate)  # minim 1s
        scores_other = []
        scores_stop = []
        step = frame // 2
        for offset in range(0, len(samples) - frame + 1, step):
            chunk = samples[offset : offset + frame].astype(np.float32) / 32768.0
            mel = logfbank(chunk, samplerate=args.sample_rate, nfilt=40, winlen=0.04, winstep=0.02)
            if mel.shape[0] < 81:
                continue
            mel = mel[:81, :].T  # (40,81)
            inp = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
            out = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
            scores_other.append(float(out[0]))
            scores_stop.append(float(out[1]))
            print(f"scores=[other={out[0]:.3f}, stop={out[1]:.3f}]")
        print(f"Peak stop score: {max(scores_stop, default=0.0):.3f}")
        print(f"Peak other score: {max(scores_other, default=0.0):.3f}")
        return

    mdl = Model(
        wakeword_models=[str(model_path)],
        inference_framework=args.framework,
        sr=args.sample_rate,
        enable_speex_noise_suppression=False,
    )
    frame = max(80, int(args.frame_size))
    scores_peak: Dict[str, float] = {}
    for offset in range(0, len(samples), frame):
        chunk = samples[offset : offset + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)), mode="constant")
        scores = mdl.predict(chunk)
        for name, score in scores.items():
            scores_peak[name] = max(scores_peak.get(name, 0.0), float(score))

    if not scores_peak:
        print("Modelul nu a returnat scoruri.")
        return

    print("Scoruri maxime OpenWakeWord:")
    for name, score in scores_peak.items():
        print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    main()
