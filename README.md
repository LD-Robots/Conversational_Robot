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
- 💬 **Stream LLM responses** for faster perceived latency
- 🧠 **Maintain conversation history** (context-aware responses)
- 🌐 **Auto web search** (compound-beta model decides when to search)
- 🤖 **Motor command integration** via tagged responses `[MOTOR:action:param]`
- 😊 **Sentiment detection** and proactive suggestions
- 💾 **TTS caching** for instant playback of common phrases

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
python -m src.server.api --host 0.0.0.0 --port 8002
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

## ⚡ Advanced Features

### LLM Streaming
Responses are streamed token-by-token for instant feedback:
```python
# In configs/llm.yaml
streaming: true  # Enables token-by-token generation
```

### Conversation History
Maintains context across multiple turns:
```yaml
# In configs/llm.yaml
history_enabled: true
max_history_turns: 2  # Keeps last 2 user/assistant exchanges
```

### Auto Web Search (Compound-Beta)
The `compound-beta` model automatically searches the web when needed:
```yaml
# In configs/llm.yaml
model: "compound-beta"  # Auto web search when lacking info
```
⚠️ Note: compound-beta takes 3-8s due to web search vs. ~1s for llama-3.1-8b-instant

### Motor Command Integration
LLM can control robot motors via tagged responses:
```
User: "Raise your left hand"
Bot: "Ok, done. [MOTOR:raise_hand:left]"
```

Supported tags:
- `[MOTOR:raise_hand:left|right]` - Raise specified hand
- `[MOTOR:wave:left|right]` - Wave hand
- `[MOTOR:nod_head]` - Nod head
- `[INTENT:question]` - Indicates user asked a question
- `[INTENT:greeting]` - Greeting detection

### TTS Caching
Common phrases are pre-synthesized and cached for instant playback (\<100ms):
```yaml
# In configs/tts.yaml
cache_enabled: true
common_phrases:
  - "Hello!"
  - "I don't understand."
  - "Let me think about that."
```

### Sentiment Detection
Detects user sentiment and provides proactive suggestions:
- Positive sentiment → Encouraging responses
- Negative sentiment → Supportive tone
- Neutral → Standard informative responses


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
python -m src.server.api --host 127.0.0.1 --port 8001
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

### Wake Words (Start Conversation)
Say any of these to wake up the robot:

| Language | Command | Model |
|----------|---------|-------|
| English | "Hello robot" | `hello_robot` |
| English | "What's up buddy" | `wots_up_bud_dee` |
| English | "Listen up" | `lis_un_up` |
| Romanian | "Bună robot" | `boona_ro_bot` |
| Romanian | "Salut robot" | `sa_loot_ro_bot` |

### Session Commands

| Command | Action | Detection Method |
|---------|--------|------------------|
| **"Goodbye robot"** | End session and return to standby | OpenWakeWord (hotword) |
| **"Bye bye"** | End session (alternative) | OpenWakeWord (hotword) |
| **"Stop robot"** | Stop current TTS playback immediately | Stop Keyword Detector (ONNX) |

### How It Works
- **Wake words**: Always listening in standby mode via OpenWakeWord
- **Goodbye**: Active only during conversation session
- **Stop**: Active only during TTS playback (requires 2 consecutive detections for reliability)

---

## 🔗 Related Files

- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [FEATURES.md](FEATURES.md) - Feature documentation
- [LIMITATIONS.md](LIMITATIONS.md) - Known limitations
