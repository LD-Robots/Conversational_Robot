# 🤖 Conversational Bot - Lista de Funcționalități

## 🎯 Prezentare Generală
Asistent vocal local, privat, bilingv (Română/Engleză) cu latență scăzută și funcționare complet offline.

---

## ✨ Funcționalități Principale

### 🎤 Wake Word Detection
- **OpenWakeWord** - Activare vocală cu "hello robot" (model ONNX local)
- **Fără API extern** - Funcționează complet offline
- **Fallback text** - Recunoaștere prin ASR dacă modelul lipsește

### 🗣️ Speech-to-Text (ASR)
- **Faster-Whisper** - Transcriere rapidă, optimizată pentru CPU
- **Detecție automată limbă** - Română sau Engleză
- **ASR Warm-up** - Pre-încărcare la boot pentru răspuns rapid

### 🧠 Language Model (LLM)
- **Ollama + Qwen 2.5** - Model local, fără conexiune internet
- **Streaming generation** - Răspunsuri în timp real
- **Conversation history** - Menține contextul (follow-up questions)
- **LLM Warm-up** - Reduce latența primei întrebări de la ~10s la ~1s

### 🔊 Text-to-Speech (TTS)
- **Piper TTS** - Sinteză vocală naturală, locală
- **Streaming TTS** - Vorbește pe măsură ce generează
- **Double buffering** - Fără micro-pauze între propoziții
- **TTS Pre-caching** - Fraze comune pre-generate (zero latency)

### 🛑 Stop Command
- **"Stop robot"** - Oprește instant vorbirea botului
- **Model ONNX dedicat** - Detectare în timp real
- **Nu întrerupe la voce normală** - Doar la comanda explicită

### 👋 Goodbye Detection
- **"Goodbye robot"** - Închide sesiunea elegant
- **OpenWakeWord model** - Detecție în timp real
- **Confirmare vocală** - "See you later!" / "Ne auzim!"

### 🌐 Suport Bilingv
- **Română și Engleză** - Detectare automată
- **TTS în limba userului** - Răspunde în limba în care întrebi
- **Switching natural** - Poți schimba limba mid-conversație

---

## ⚡ Optimizări de Performanță

| Optimizare | Beneficiu |
|------------|-----------|
| **LLM Warm-up** | Prima întrebare ~1s în loc de ~10s |
| **ASR Warm-up** | Transcriere mai rapidă |
| **TTS Pre-cache** | Confirmări instant |
| **Double Buffer** | Vorbire fluidă |
| **Backchannel** | "Un moment..." când durează |

---

## 🔧 Arhitectură Tehnică

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Microfon   │───▶│ OpenWakeWord│───▶│   Sesiune   │
│   (ec_mic)  │    │ "hello robot"│    │   Activă    │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
     ┌───────────────────────────────────────┘
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Record    │───▶│ Faster-     │───▶│   Ollama    │
│   + VAD     │    │ Whisper ASR │    │  Qwen 2.5   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
     ┌────────────────────────────────────────┘
     ▼
┌─────────────┐    ┌─────────────┐
│  Piper TTS  │───▶│   Speaker   │
│  (stream)   │    │ (ec_speaker)│
└─────────────┘    └─────────────┘
```

---

## 🔒 Privacitate & Securitate

- ✅ **100% Local** - Nicio dată nu părăsește dispozitivul
- ✅ **Fără cloud** - Nu necesită conexiune internet
- ✅ **Fără API keys externe** - OpenWakeWord, Piper, Ollama - toate gratuite
- ✅ **Open source** - Cod verificabil

---

## 📊 Metrici în Timp Real

Dashboard disponibil la `http://localhost:9108/vitals`:
- Round-trip time (întrebare → răspuns complet)
- ASR latency
- LLM time-to-first-token
- Sesiuni active
- Erori

---

## 🎬 Demo Flow Recomandat

1. **"Hello robot"** → Activare
2. **"Ce faci?"** → Răspuns în română
3. **"What's the capital of France?"** → Răspuns în engleză
4. **"And Germany?"** → Follow-up (conversation history)
5. **"Stop robot"** → Oprește vorbirea
6. **"Goodbye robot"** → Închide sesiunea
