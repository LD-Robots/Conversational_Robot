# 🤖 Conversational Bot - Client-Server Architecture

A bilingual (Romanian/English) voice-controlled conversational bot with a **client-server architecture** that allows distributing processing across multiple machines.

---

## 📋 Overview

This project implements a voice assistant that can:
- 🎤 Listen for wake words ("hello robot")
- 🧏 Transcribe speech to text (ASR)
- 🧠 Generate intelligent responses (LLM)
- 🔊 Speak responses naturally (TTS)
- 🛑 Handle interruptions ("stop robot", "goodbye robot")

### Architecture Modes

|        Mode       |             Description                   |
|-------------------|-------------------------------------------|
| **Local**         | All processing on one machine             |
| **Client-Server** | Audio I/O on client, processing on server |

```
┌─────────────────────────┐         ┌─────────────────────────┐
│       CLIENT            │   HTTP  │        SERVER           │
│                         │◄───────►│                         │
│  🎤 Audio Capture       │         │  🧏 ASR (Whisper)       │
│  👂 Wake Word Detection │         │  🧠 LLM (Groq/Ollama)   │
│  🔊 Audio Playback      │         │  🗣️ TTS (Edge TTS)      │
│  🛑 Stop Keyword        │         │                         │
└─────────────────────────┘         └─────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Ubuntu 22.04/24.04 (or compatible Linux)
- Microphone and speakers
- Internet connection (for Groq LLM and Edge TTS)

### Installation

```bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key for Groq
echo "GROQ_API_KEY=your_key_here" > .env
```

### Running in Local Mode

```bash
# Set configs to local mode (in configs/*.yaml):
# mode: local

source .venv/bin/activate
LOG_LEVEL=INFO python -m src.app
```

### Running in Client-Server Mode

**Terminal 1 - Server:**
```bash
source .venv/bin/activate
python -m src.server.api --host 0.0.0.0 --port 8001
```

**Terminal 2 - Client:**
```bash
# Set configs to remote mode (in configs/*.yaml):
# mode: remote
# remote_host: "localhost"  # or server IP
# remote_port: 8001

source .venv/bin/activate
LOG_LEVEL=INFO python -m src.app
```

---

## 📁 Project Structure

```
Conversational_Bot/
├── configs/                    # Configuration files
│   ├── asr.yaml               # ASR settings (Whisper)
│   ├── llm.yaml               # LLM settings (Groq/Ollama)
│   ├── tts.yaml               # TTS settings (Edge TTS)
│   ├── audio.yaml             # Audio & barge-in settings
│   └── wake.yaml              # Wake word settings
│
├── src/
│   ├── app.py                 # 🎯 Main client application
│   │
│   ├── server/                # 🖥️ Server API
│   │   ├── api.py             # Flask REST endpoints
│   │   └── __init__.py
│   │
│   ├── asr/                   # 🧏 Speech-to-Text
│   │   ├── interface.py       # ASRInterface, LocalASR, RemoteASR
│   │   ├── engine_faster.py   # Faster-Whisper implementation
│   │   └── __init__.py        # Factory: make_asr()
│   │
│   ├── llm/                   # 🧠 Language Model
│   │   ├── interface.py       # LLMInterface, LocalLLM, RemoteLLM
│   │   ├── engine.py          # Groq/Ollama/OpenAI implementation
│   │   └── __init__.py        # Factory: make_llm()
│   │
│   ├── tts/                   # 🔊 Text-to-Speech
│   │   ├── interface.py       # TTSInterface, LocalTTS, RemoteTTS
│   │   ├── edge_backend.py    # Microsoft Edge TTS
│   │   ├── engine.py          # Piper/pyttsx3 fallback
│   │   └── __init__.py        # Factory: make_tts()
│   │
│   ├── audio/                 # 🎤 Audio processing
│   │   ├── input.py           # Audio recording
│   │   ├── barge.py           # Barge-in detection
│   │   ├── vad.py             # Voice Activity Detection
│   │   └── stop_keyword_detector.py
│   │
│   ├── wake/                  # 👂 Wake word detection
│   │   └── openwakeword_engine.py
│   │
│   ├── core/                  # ⚙️ Core utilities
│   │   ├── config.py          # Config loader
│   │   ├── logger.py          # Logging setup
│   │   └── fast_exit.py       # Goodbye detection
│   │
│   └── telemetry/             # 📊 Metrics
│       └── metrics.py         # Prometheus metrics
│
├── voices/                    # ONNX voice models
│   ├── hello_robot.onnx       # Wake word model
│   ├── goodbye_robot.onnx     # Goodbye detection
│   └── stop_keyword.onnx      # Stop command model
│
├── models/                    # ASR models (Whisper)
├── tools/                     # Utility scripts
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Configuration

### Client-Server Mode Settings

Each module (ASR, LLM, TTS) can be configured independently in their YAML files:

```yaml
# configs/asr.yaml
mode: remote                # local | remote
remote_host: "192.168.1.100"
remote_port: 8001
remote_timeout: 30.0
```

### Key Configuration Files

|          File        |                 Description            |
|----------------------|----------------------------------------|
| `configs/asr.yaml`   | Whisper model size, language, mode     |
| `configs/llm.yaml`   | Provider (groq/ollama), model, prompts |
| `configs/tts.yaml`   | Voice selection, caching, mode         |
| `configs/audio.yaml` | VAD, barge-in thresholds, stop keyword |
| `configs/wake.yaml`  | Wake phrases, OpenWakeWord settings    |

---

## 🔌 Server API Endpoints

|       Endpoint      | Method |         Description             |
|---------------------|--------|---------------------------------|
| `/health`           |   GET  | Server health check             |
| `/transcribe`       |   POST | Transcribe audio (WAV → text)   |
| `/transcribe_ro_en` |   POST | Bilingual transcription (RO/EN) |
| `/generate`         |   POST | Generate LLM response           |
| `/generate_stream`  |   POST | Stream LLM tokens               |
| `/synthesize`       |   POST | Synthesize speech (text → MP3)  |

---

## 🔧 Technologies Used

|   Component   |              Technology              |
|---------------|--------------------------------------|
| **ASR**       | Faster-Whisper (Whisper optimized)   |
| **LLM**       | Groq Cloud (llama-3.3-70b) or Ollama |
| **TTS**       | Microsoft Edge TTS (Neural voices)   |
| **Wake Word** | OpenWakeWord (custom ONNX)           |
| **Server**    | Flask (REST API)                     |
| **Audio**     | sounddevice, WebRTC VAD              |

---

## 🌐 Deployment on Two Machines

### Step 1: Clone on both machines

```bash
# On both laptops:
git clone https://github.com/Delia63/Conversational_Robot.git
cd Conversational_Robot/Conversational_Bot
git checkout client-server
pip install -r requirements.txt
```

### Step 2: Configure server (Laptop 2)

```bash
# Start server listening on all interfaces
python -m src.server.api --host 0.0.0.0 --port 8001
```

### Step 3: Configure client (Laptop 1)

Edit `configs/asr.yaml`, `configs/llm.yaml`, `configs/tts.yaml`:
```yaml
mode: remote
remote_host: "192.168.1.X"  # Replace with Laptop 2 IP
remote_port: 8001
```

```bash
# Start client
LOG_LEVEL=INFO python -m src.app
```

---

## 📊 Performance Metrics

|    Metric       |         Typical Value      |
|-----------------|----------------------------|
| ASR Latency     | ~3-5s (Whisper small, CPU) |
| LLM First Token | ~200-300ms (Groq)          |
| Round-trip      | ~2-3s                      |
| TTS Cache Play  | <100ms                     |

Access metrics at: `http://localhost:9108/vitals`

---

## 🎯 Voice Commands

|     Command     |        Action             |
|-----------------|---------------------------|
| "Hello robot"   | Wake up and start listenin|
| "Goodbye robot" | End session               |
| "Stop robot"    | Stop current TTS playback |

---

## 📝 License

This project is for educational and research purposes.

---

## 🔗 Related Files

- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [FEATURES.md](FEATURES.md) - Feature documentation
- [LIMITATIONS.md](LIMITATIONS.md) - Known limitations
