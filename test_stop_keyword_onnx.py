import numpy as np
import soundfile as sf
import resampy
import torch
import torchaudio
import onnxruntime as ort

MODEL_PATH = "voices/stop_keyword.onnx"
WAV_PATH = "stop_robot.wav"

SR = 16000
FRAME = SR  # 1 secundă

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_mels=40,
)
db_transform = torchaudio.transforms.AmplitudeToDB()

def pad_or_trim(x, target_len=FRAME):
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        pad = target_len - len(x)
        return np.pad(x, (0, pad))
    return x

def featurize(chunk):
    # chunk: numpy [T]
    t = torch.from_numpy(chunk[None, :])  # [1, T] = [channels, time]
    mel = mel_transform(t)                # [1, 40, 81]
    mel_db = db_transform(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    mel_db = mel_db.unsqueeze(0)          # [1, 1, 40, 81] - adăugăm batch dim
    feats = mel_db.numpy().astype(np.float32)
    return feats

audio, sr = sf.read(WAV_PATH)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo -> mono
if sr != SR:
    audio = resampy.resample(audio, sr, SR)
audio = audio.astype(np.float32)

stop_scores = []
other_scores = []

for i in range(0, len(audio) - FRAME + 1, FRAME // 2):
    chunk = audio[i:i+FRAME]
    chunk = pad_or_trim(chunk, FRAME)
    feats = featurize(chunk)
    ort_inputs = {input_name: feats}
    logits = session.run(None, ort_inputs)[0][0]  # [2]
    other, stop = logits
    stop_scores.append(float(stop))
    other_scores.append(float(other))
    print(f"chunk {i/SR:.2f}-{(i+FRAME)/SR:.2f}s -> logits: other={other:.3f}, stop={stop:.3f}")

if stop_scores:
    print("Peak stop logit:", max(stop_scores))
    print("Peak other logit:", max(other_scores))
else:
    print("No chunks processed")
