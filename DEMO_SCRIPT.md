# 🎬 Demo Script - Conversational Bot

## 📋 Setup (before demo)

```bash
# 1. Activate virtual environment
cd /home/delia/Conversational_Robot/Conversational_Bot
source .venv/bin/activate

# 2. Verify Ollama is running
ollama list  # should show qwen2.5:3b

# 3. Start the bot
LOG_LEVEL=INFO LOG_DIR=logs python -m src.app
```

---

## 🎯 Demo Scenario (5-7 minutes)

### **ACT 1: Wake Word & First Conversation** (1 min)

**YOU:** *(speak clearly)* "**Hello robot**"

**BOT:** "Yes, I'm listening."

**YOU:** "What is the capital of France?"

**BOT:** "The capital of France is Paris..."

> 💡 **What you demonstrate:** Wake word detection with OpenWakeWord, ASR with Faster-Whisper, LLM response

---

### **ACT 2: Conversation History (Follow-up)** (1 min)

**YOU:** "And what about Germany?"

**BOT:** "The capital of Germany is Berlin..."

> 💡 **What you demonstrate:** Context retention - the bot knows we're talking about capitals without repeating the question

---

### **ACT 3: Curiosity Question** (1 min)

**YOU:** "Tell me something interesting about Madrid."

**BOT:** *(responds with a fun fact)*

**YOU:** "Tell me another one."

**BOT:** *(shares another curiosity)*

> 💡 **What you demonstrate:** Multi-turn conversation, context awareness

---

### **ACT 4: Stop Keyword** (30 sec)

**YOU:** "Explain quantum physics in detail."

**BOT:** *(starts a long explanation)*

**YOU:** "**Stop robot**"

**BOT:** *(TTS stops immediately)*

> 💡 **What you demonstrate:** Stop keyword with ONNX model - instant detection and interruption

---

### **ACT 5: Session Close** (30 sec)

**YOU:** "**Goodbye robot**"

**BOT:** "Goodbye! Have a great day!"

*(Session closes, returns to standby)*

> 💡 **What you demonstrate:** Graceful exit with OpenWakeWord goodbye detection

---

## 📊 Technical Points to Mention

| Feature | Technology |
|---------|------------|
| Wake Word | OpenWakeWord (ONNX, local) |
| Speech-to-Text | Faster-Whisper (local, CPU) |
| LLM | Ollama + Qwen 2.5 3B (local) |
| Text-to-Speech | Piper TTS (local) |
| Stop Detection | Custom ONNX classifier |
| Echo Cancel | PulseAudio AEC |

**Everything runs LOCALLY** - no internet connection, no external API keys!

---

## ⚠️ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot doesn't hear "hello robot" | Check microphone volume in pavucontrol |
| Echo / hears itself | Select `ec_mic` input in pavucontrol |
| LLM is slow | Normal on CPU, would be faster on GPU |
| No response | Verify Ollama is running: `ollama list` |

---

## 🎤 Presentation Tips

1. **Speak clearly and slowly** - ASR works better with clear speech
2. **Pause after wake word** - wait for "Yes, I'm listening" before continuing
3. **Low ambient noise** - find a quiet spot
4. **Backup plan** - if something fails, mention it's WIP and continue
