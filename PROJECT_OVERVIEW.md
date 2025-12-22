# 🤖 Conversational Bot - Project Overview

**Status**: Funcțional și optimizat  
**Branch**: `main`  
**Dată**: 22 Decembrie 2024

---

## 📋 Descriere Generală

Un bot conversațional vocal bilingv (Română/Engleză) care poate:
- Asculta comenzi vocale și le transcrie
- Înțelege contextul și genera răspunsuri inteligente
- Vorbi înapoi cu voci naturale
- Detecta când utilizatorul vrea să întrerupă (barge-in, stop keyword)

---

## 🏗️ Arhitectură

```
┌─────────────────────────────────────────────────────────────────┐
│                         STANDBY MODE                            │
│   🎧 OpenWakeWord ascultă: "hello robot" → activare sesiune    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONVERSATION LOOP                          │
│                                                                 │
│  🎤 Record Audio ──▶ 🧏 ASR (Whisper) ──▶ 🧠 LLM (Groq)        │
│        │                    │                    │              │
│        │                    │                    ▼              │
│        │                    │         📝 Text Response          │
│        │                    │                    │              │
│        │                    │                    ▼              │
│        ◀────────────────────┴──────────── 🔊 TTS (Edge)        │
│                                                                 │
│  🛑 Stop Keyword / "goodbye robot" → exit loop                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Componente Principale

### 1. Wake Word Detection
| Aspect | Detalii |
|--------|---------|
| **Engine** | OpenWakeWord |
| **Wake Phrase** | "hello robot" |
| **Goodbye Phrase** | "goodbye robot" |
| **Model** | `voices/hello_robot.onnx` |

### 2. Speech-to-Text (ASR)
| Aspect | Detalii |
|--------|---------|
| **Engine** | faster-whisper (Whisper) |
| **Model** | `medium` (769MB) |
| **Compute** | CPU, int8 quantized |
| **Latență** | ~7-8s pentru română |
| **Limbi** | Auto-detect RO/EN |

### 3. Language Model (LLM)
| Aspect | Detalii |
|--------|---------|
| **Provider** | Groq Cloud |
| **Model** | llama-3.3-70b-versatile |
| **First Token** | ~180-300ms |
| **Streaming** | Da |
| **History** | 5 turns |

### 4. Text-to-Speech (TTS)
| Aspect | Detalii |
|--------|---------|
| **Backend** | Edge TTS (Microsoft Neural) |
| **Voce EN** | Sonia (en-GB-SoniaNeural) |
| **Voce RO** | Emil (ro-RO-EmilNeural) |
| **Cache** | 6 fraze pre-generate |
| **Playback** | ffplay (subprocess) |

### 5. Barge-in & Stop
| Aspect | Detalii |
|--------|---------|
| **Stop Keyword** | Model ONNX custom |
| **Voice Detection** | WebRTC VAD |
| **Thresholds** | RMS, ZCR, highpass filter |

---

## 📁 Structura Proiectului

```
Conversational_Bot/
├── src/
│   ├── app.py              # Entry point principal
│   ├── asr/                # Speech-to-Text
│   │   ├── engine_faster.py  # Whisper via faster-whisper
│   │   └── __init__.py       # Factory pentru ASR
│   ├── llm/
│   │   └── engine.py       # Ollama, Groq, OpenAI support
│   ├── tts/
│   │   ├── engine.py       # TTS factory (Piper, Edge, pyttsx3)
│   │   └── edge_backend.py # Edge TTS implementation
│   ├── wake/               # Wake word detection
│   ├── audio/              # Audio capture, playback, barge-in
│   ├── core/               # Config, logging, states
│   ├── telemetry/          # Prometheus metrics
│   └── utils/              # Helpers
├── configs/
│   ├── asr.yaml            # ASR settings
│   ├── llm.yaml            # LLM settings + system prompt
│   ├── tts.yaml            # TTS settings
│   ├── audio.yaml          # Audio + barge-in settings
│   └── wake.yaml           # Wake word settings
├── voices/                 # Voice models (Piper, wake word)
├── models/                 # ASR models (Vosk)
└── data/cache/             # Temporary audio files
```

---

## ⚙️ Configurații Curente

### ASR (`configs/asr.yaml`)
```yaml
provider: faster
model_size: medium
compute_type: int8
device: cpu
```

### LLM (`configs/llm.yaml`)
```yaml
provider: groq
model: llama-3.3-70b-versatile
max_tokens: 120
history_enabled: true
```

### TTS (`configs/tts.yaml`)
```yaml
backend: edge
edge_voice_en: en-GB-SoniaNeural
edge_voice_ro: ro-RO-EmilNeural
```

---

## 📊 Metrici de Performanță

| Metrică | Valoare Tipică |
|---------|----------------|
| **ASR Latency** | ~7-8s (Whisper medium, CPU) |
| **LLM First Token** | ~180-300ms (Groq) |
| **Round-trip** | ~2.5-3s |
| **TTS Cache Play** | <100ms |

---

## 🔧 Tehnologii Utilizate

### Python Packages
| Package | Versiune | Rol |
|---------|----------|-----|
| faster-whisper | 1.0.3 | ASR |
| groq | 1.0.0 | LLM API |
| edge-tts | latest | TTS |
| openwakeword | 0.6.0 | Wake word |
| sounddevice | 0.4.6 | Audio I/O |
| webrtcvad | 2.0.10 | Voice Activity |
| torch | ≥2.0 | ML backend |
| onnxruntime | 1.18.1 | Inference |

### Servicii Externe
- **Groq Cloud** - LLM inference (API key necesar)
- **Microsoft Edge TTS** - Sinteză vocală (gratuit, necesită internet)

### Dependențe Sistem
- Python 3.11+
- ffplay (pentru playback audio)
- PulseAudio/PipeWire (pentru captură audio)

---

## 🚀 Cum să Pornești

```bash
# Activează virtual environment
source .venv/bin/activate

# Setează API key-ul Groq în .env
echo "GROQ_API_KEY=your_key_here" > .env

# Pornește botul
LOG_LEVEL=INFO python -m src.app
```

---

## 📝 Funcționalități Cheie

1. **Wake Word** - Activare hands-free cu "hello robot"
2. **Bilingv** - Auto-detect și răspuns în RO/EN
3. **Streaming TTS** - Răspunsul începe înainte de a termina generarea
4. **Stop Keyword** - Oprește TTS cu "stop robot"
5. **History** - Ține minte contextul conversației (5 turns)
6. **Fallback Responses** - Mesaje inteligente pentru erori
7. **Metrics** - Prometheus endpoint pentru monitorizare
8. **Cache TTS** - Fraze comune pre-generate

---

## 🔮 Limitări Cunoscute

- **ASR Latency**: ~7-8s pe CPU pentru română (Whisper medium)
- **Requires Internet**: Edge TTS și Groq necesită conexiune
- **English Wake Word**: "hello robot" funcționează doar în engleză
- **No GPU Acceleration**: Toate modelele rulează pe CPU

---

## 📈 Îmbunătățiri Posibile

1. **Google Speech API** pentru ASR mai rapid în română
2. **GPU (CUDA)** pentru Whisper local rapid
3. **Server remote** cu GPU pentru ASR
4. **Wake word în română** (necesită antrenare model custom)
