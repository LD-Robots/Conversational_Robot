#!/usr/bin/env python3
"""Test OpenWakeWord model on a WAV file - shows scores for detection."""
import sys
import numpy as np
from scipy.io import wavfile
from pathlib import Path

def test_model(model_path: str, wav_path: str):
    from openwakeword.model import Model
    
    model_path = str(Path(model_path).resolve())
    print(f"🔧 Loading model: {model_path}")
    
    # Load model - for custom models trained via Colab
    model = Model(wakeword_models=[model_path], inference_framework="onnx")
    
    print(f"   Available models: {list(model.models.keys())}")
    
    print(f"\n🎵 Loading audio: {wav_path}")
    sample_rate, audio = wavfile.read(wav_path)
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Duration: {len(audio)/sample_rate:.2f}s")
    print(f"   Samples: {len(audio)}")
    
    # Process in 80ms chunks (1280 samples @ 16kHz)
    chunk_size = 1280
    max_scores = {}
    
    print(f"\n📊 Processing audio in {chunk_size}-sample chunks...")
    
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        predictions = model.predict(chunk)
        
        for name, score in predictions.items():
            if name not in max_scores:
                max_scores[name] = 0.0
            if score > max_scores[name]:
                max_scores[name] = score
            if score > 0.3:  # Show scores above 0.3
                time_s = i / sample_rate
                print(f"   [{time_s:.2f}s] {name}: {score:.4f}")
    
    print(f"\n✅ RESULTS:")
    for name, score in max_scores.items():
        status = "✓ DETECTED" if score >= 0.5 else "✗ not detected"
        print(f"   {name}: max={score:.4f} {status}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_wake_model.py <model.onnx> <audio.wav>")
        print("\nExamples:")
        print("  python test_wake_model.py voices/hello_robot.onnx test.wav")
        print("  python test_wake_model.py voices/buna.onnx buna.wav")
        sys.exit(1)
    
    test_model(sys.argv[1], sys.argv[2])
