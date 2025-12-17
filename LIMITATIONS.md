# 🚧 Project Limitations

## Hardware & Performance

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **CPU-only inference** | LLM responses take 2-10+ seconds | Use GPU with CUDA for faster inference |
| **RAM usage** | Models require 4-8GB RAM | Use smaller models (3B instead of 7B) |
| **Single speaker** | Only one user can interact at a time | By design for personal assistant |

---

## Speech Recognition (ASR)

| Limitation | Impact |
|------------|--------|
| **Background noise sensitivity** | ASR accuracy drops in noisy environments |
| **Accent handling** | May struggle with strong regional accents |
| **Short utterances** | Words under 0.35s are ignored (anti-noise filter) |
| **Romanian transcription** | Whisper sometimes misinterprets Romanian as other languages |
| **Homophone confusion** | "stop robot" vs sound-alike phrases may trigger false positives |

---

## Language Model (LLM)

| Limitation | Impact |
|------------|--------|
| **Romanian quality** | Qwen 2.5 3B has limited Romanian vocabulary and grammar |
| **Factual accuracy** | May hallucinate facts (e.g., "București is capital of France") |
| **Context window** | Limited conversation history (~10 turns) |
| **No internet access** | Cannot provide real-time information (weather, news, etc.) |
| **No memory persistence** | Forgets everything when session ends |
| **Response truncation** | Long responses may be cut off by max_tokens limit |

---

## Text-to-Speech (TTS)

| Limitation | Impact |
|------------|--------|
| **Robotic voice** | Piper TTS sounds less natural than cloud services |
| **Intonation** | Limited emotional expression in voice |
| **Multi-language mixing** | Cannot seamlessly switch languages mid-sentence |
| **Pronunciation** | May mispronounce proper nouns or technical terms |

---

## Wake Word & Barge-in

| Limitation | Impact |
|------------|--------|
| **Fixed wake phrase** | Only "hello robot" is supported |
| **Echo sensitivity** | Speaker feedback may trigger false barge-in |
| **Stop keyword model** | Custom ONNX model may have false positives/negatives |
| **Requires quiet environment** | Best performance in low-noise settings |

---

## System & Integration

| Limitation | Impact |
|------------|--------|
| **Linux only** | PulseAudio AEC setup is Linux-specific |
| **No visual feedback** | Text-only interface, no GUI |
| **Single language session** | Language detection per-turn, not per-session |
| **No user authentication** | Cannot distinguish between different users |
| **No external APIs** | Cannot integrate with smart home, calendar, etc. |

---

## Known Issues

1. **LLM sometimes responds in wrong language** - Even with Romanian input, may reply in English
2. **First response is slow** - Cold start takes 5-15 seconds on CPU
3. **Barge-in false positives** - Echo from speakers may trigger interruption
4. **ASR misses quiet speech** - Very soft voice may not be detected
5. **No graceful degradation** - If Ollama is down, bot crashes

---

## Comparison with Cloud Solutions

| Feature | This Bot | Cloud (Alexa/Google) |
|---------|----------|---------------------|
| Privacy | ✅ 100% local | ❌ Data sent to servers |
| Response time | ❌ 2-10s on CPU | ✅ <1s |
| Voice quality | ⚠️ Robotic | ✅ Natural |
| Languages | ⚠️ EN/RO only | ✅ 50+ languages |
| Knowledge | ❌ Static, no internet | ✅ Real-time info |
| Cost | ✅ Free (open source) | ⚠️ Subscription/device |
