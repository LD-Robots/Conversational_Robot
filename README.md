# 🧠 Conversational Bot

Private, local, low-latency voice assistant with hotword detection, ASR, **streaming LLM → streaming TTS**, barge-in, and a tidy `/vitals` dashboard.

---

## ✨ What’s implemented (and how)

* **Wake word with safe fallback** — Porcupine hotword; if it’s missing or fails, the app switches to **text-based wake matching** without crashing.
* **ASR with clean endpointing** — Faster-Whisper tuned for short turns; **standby** listens in tight windows; **active sessions** auto-detect RO/EN (standby favors EN for reliable hotwords).
* **Streaming LLM → streaming TTS** — Real-time token streaming to speech; **time-to-first-token (TTFT)** is measured so replies feel snappy.
* **Latency backchannel** — If TTFT exceeds ~2s, the bot plays "One moment…" / "Un moment…" so you know it's working.
* **Audio hygiene** — System echo-cancel (AEC), noise suppression, high-pass filter; **AGC off** to avoid noise pumping & false VAD triggers.
* **PyTorch stop keyword** — custom ONNX model (`audio.stop_keyword`) monitors the mic only while TTS talks and instantly cuts playback when you say "stop robot".
* **No accidental "pa…" exits** — Session closes **only** on exact goodbyes (e.g., "ok bye", "gata", "la revedere").
* **Observability** — Prometheus counters + a simple `/vitals` page for round-trip, ASR, TTFT, sessions, turns, errors.
* **Double buffer for seamless TTS** — Prevents micro-pauses when the bot speaks; while buffer A plays, buffer B synthesizes the next chunk, then they alternate continuously.
* **English <> Romanian** — Improved command & QA flow in English while keeping full Romanian support.
* **Honest fallback** — If the bot doesn’t know, it says so (“I’m not sure about that yet, but I can look it up if you’d like.”).
* **Graceful CTRL+C shutdown** — One keystroke stops TTS, flushes buffers, dumps a metrics snapshot, and closes all background listeners.

### 🚀 Performance Optimizations

* **LLM Warm-up** — At boot, performs a dummy request to load the model into RAM, reducing first-query latency from ~6-10s to ~0.3-2s.
* **ASR Warm-up** — Runs a silent dummy transcription at startup to fully load Faster-Whisper into memory, speeding up the first real transcription.
* **TTS Pre-caching** — Pre-generates WAV files for common phrases (acknowledgements, fillers) at boot for zero-latency playback.
* **Conversation History** — Maintains context throughout the session, allowing follow-up questions like "And Germany?" after asking about France.
* **Fallback Responses** — Configurable error messages for timeout, connection errors, and empty responses instead of crashing.
* **Sentiment Detection** — LLM adapts responses based on user's emotional state (frustrated, curious, confused).
* **Proactive Suggestions** — Bot offers helpful follow-up suggestions when appropriate.

---

## 🔧 Practical setup for users (do this)

1. **Select the echo-cancelled mic** 
   Use the `ec_mic` input (see **Linux audio** + **Audio routing** below).

2. **Tune thresholds for your room**
   - `min_speech_duration`: **1.0–1.2s** (utterances shorter than this are ignored)
   - `silence_to_end`: **1200–1500 ms** (only for *active* session end)
   - Keep **AGC off** in the OS/driver and inside AEC if exposed.

3. **Keys & env**
   - Put secrets in `.env` (e.g., `PICOVOICE_ACCESS_KEY=...`).
   - Activating a venv **does not** read `.env`. Either:
     - use `python-dotenv` inside the app, **or**
     - `export $(grep -v '^#' .env | xargs)` before `python -m src.app`.

4. **Run with structured logs**
   ```bash
   LOG_LEVEL=INFO LOG_DIR=logs python -m src.app
   ```
   Press **CTRL+C once** to exit cleanly — it stops TTS, flushes buffers, dumps a metrics snapshot, and closes all background listeners (no need for `pkill`).

5. **(Optional) Wake Hotword** 
   Picovoice Porcupine for “hello robot”; if missing, text fallback is used.

6. **Stop command (PyTorch detector)** 
   `audio.stop_keyword` loads `voices/stop_keyword.onnx` (other vs stop classes) and runs only while TTS is speaking. Tune `logit_margin`, `prob_threshold`, or `hits_required` if you need stricter detection.
   Ajustează `tts.backchannel.delay_ms/phrase_*` dacă vrei să schimbi filler-ul „One moment…” care acoperă latențele mari la TTFT.

### Backchannel (TTFT filler)

- Configure it in `configs/tts.yaml` (`backchannel.enabled`, `delay_ms`, `phrase_en`, `phrase_ro`).
- In `src/app.py` we track TTFT with a `threading.Event`; if no token arrives within the threshold we play `tts.say("One moment...")` (or “Un moment…” for Romanian) before streaming the real reply.
- The backchannel respects FastExit and stop signals, so it never fights barge-in or manual cancels.
- This hides long LLM latencies (e.g., large models on CPU) without altering the rest of the speech pipeline.

7. **Route audio correctly (AEC)** ➜ see **🔊 Audio routing (AEC) & pavucontrol** 
   TTS → `Echo-Cancel Sink`, Microphone → `Echo-Cancel Source`. Verify and adjust with pavucontrol.

---

## 🧩 Mini flow (pipeline)

**Standby & Wake** → (Porcupine **or** text fallback) 
→ **Acknowledgement** (“Yes, I’m listening.” / “Da, te ascult.”) 
→ **Record & endpoint** (VAD on silence; AEC + NS + HPF; AGC off) 
→ **ASR** (Faster-Whisper; session auto RO/EN; standby favors EN) 
→ **LLM** (streamed generation; **strict-facts** mode to reduce hallucinations) 
→ **TTS** (streamed **sentence chunks**) 
→ **Double buffer** (A plays, B synthesizes; swap) 
→ **Barge-in** (if the user speaks, TTS stops; return to listening) 
→ **Session end** (idle timeout **or** exact-match goodbye)

---

## 🎙️ Audio Architecture (AEC explained)

**Goal:** prevent the bot’s own TTS from being mis-detected as user speech.

**How:** WebRTC AEC uses an **adaptive filter** to estimate the **echo path** (far-end playback → what the mic would hear) and subtracts it from the mic stream. It adapts in real time.

**Extra guards we use:**
* **Exact-match goodbye only** (no partial "pa..." exits).
* **Voice-only gating**: prioritize voiced segments for barge-in (reduces knocks/claps).

---

## 🧪 Biggest build obstacles (and fixes)

* **Echo loop (bot hears itself)** → fixed with **system AEC** + selecting `ec_mic`, AGC off.
* **False exits on “pa…”** → fixed via **exact-match goodbyes** only.
* **TTS micro-pauses** → fixed with **double buffering**.
* **Noise-triggered barge-in** → improved by **voiced-only gating** and higher minimum speech duration.

> **BIGGEST OBSTACLE — reliable barge-in**: now solid with **Cobra VAD**. It also works *without* Picovoice (with WebRTC VAD + thresholds), but Cobra is more robust.

---

## 🧰 Linux audio: create echo-cancel devices (PulseAudio / PipeWire)

> Many modern distros run **PipeWire** with a PulseAudio compatibility layer. The commands below work in both setups if the PulseAudio modules are available.

```bash
# 1) Show current default sink/source
pactl info | sed -n -e 's/^Default Sink: /Default Sink: /p' -e 's/^Default Source: /Default Source: /p'

# 2) Unload any old echo-cancel (ignore errors if not loaded)
pactl unload-module module-echo-cancel 2>/dev/null || true

# 3) Load WebRTC echo-cancel on defaults
DEFAULT_SINK="$(pactl info | awk -F': ' '/Default Sink/{print $2}')"
DEFAULT_SOURCE="$(pactl info | awk -F': ' '/Default Source/{print $2}')"

pactl load-module module-echo-cancel \
  aec_method=webrtc \
  aec_args="analog_gain_control=0 digital_gain_control=0" \
  use_master_format=1 \
  sink_master="$DEFAULT_SINK" \
  source_master="$DEFAULT_SOURCE" \
  sink_name=ec_speaker \
  source_name=ec_mic

# 4) Make the echo-cancelled mic default
pactl set-default-source ec_mic

# 5) Verify
pactl list short sources | grep -Ei 'ec_mic|echo|cancel'
pactl list short sinks   | grep -Ei 'ec_speaker|echo|cancel'
```

---

# 🔊 Audio routing (AEC) & pavucontrol

**Target:** route **TTS → `ec_speaker`** and **Mic → `ec_mic`** so AEC has the correct playback reference and barge-in won’t trigger on your own TTS.

## 1) Install pavucontrol (Ubuntu 22.04/24.04)
```bash
sudo apt update
sudo apt install -y pavucontrol pulseaudio-utils libwebrtc-audio-processing1
```

## 2) Run the app with forced routing
```bash
# (optional) load .env secrets
test -f .env && export $(grep -v '^#' .env | xargs)

# launch using AEC devices
PULSE_SINK=ec_speaker PULSE_SOURCE=ec_mic LOG_LEVEL=INFO LOG_DIR=logs \
  ./.venv/bin/python -m src.app
```

## 3) Verify & adjust in pavucontrol (GUI)
```bash
pavucontrol &
```
- **Playback**: for the *python* process (TTS), choose **`Echo-Cancel Sink`** (ec_speaker).
- **Recording**: for the *python* process (capture), choose **`Echo-Cancel Source`** (ec_mic).
- Recommended volumes: `ec_speaker` **60–65%**, `ec_mic` **100–120%** (AGC off).

> If `ec_speaker` / `ec_mic` don’t appear in the dropdown, re-run **Create echo-cancel devices** and reopen pavucontrol.
> Once routing looks correct, stop the bot with `CTRL+C` and immediately run `tools/calibrate_audio.py` (see **Calibrate room thresholds**) so the thresholds match this setup.

## 4) Quick CLI checks
```bash
# show defaults
pactl info | sed -n -e 's/^Default Sink: //p' -e 's/^Default Source: //p'

# ec_mic becomes RUNNING while the app is listening
pactl list short sources | grep ec_mic
```

## 5) Troubleshooting
- **“No such entity” when setting volume on ec_*:** the AEC devices aren’t created — repeat the **Linux audio** section.
- **Barge-in during TTS:** ensure in pavucontrol that TTS → `ec_speaker`, mic → `ec_mic`. Lower `ec_speaker` to ~60% and (temporarily) set in `configs/audio.yaml`:
  ```yaml
  barge_allow_during_tts: false
  ```
  Optionally raise thresholds:
  ```yaml
  barge_min_voice_ms: 1000-1500
  barge_min_rms_dbfs: -20..-16
  barge_highpass_hz: 240
  ```

---

## 🎛️ Calibrate room thresholds (`tools/calibrate_audio.py`)

Use this wizard whenever you change speakers, room layout, or microphone gain so `configs/audio.yaml` reflects your real echo level.

1. **Route once, reuse everywhere.** Launch the bot briefly, open `pavucontrol`, and set the `python` playback stream to `Echo-Cancel Sink` (~60–65 % volume) and the recording stream to `Echo-Cancel Source` (100 %). Stop the bot with a single `CTRL+C`.
2. **Run the wizard (≈30 s tone).**
   ```bash
   PULSE_SOURCE=ec_mic PULSE_SINK=ec_speaker python tools/calibrate_audio.py --duration 30
   ```
   Stay silent while the tone plays; watch `pavucontrol` if you want to confirm the routing.
3. **Apply the suggestions.** At the end you’ll get lines such as:
   ```
   barge_min_rms_dbfs: 24.3
   barge_highpass_hz: 200
   ```
   Copy them into `configs/audio.yaml` (keep AGC off). These numbers are derived from the measured speaker leak, so barge-in triggers only when real speech is present.
4. **Re-run after major changes.** If you move the robot, change speaker volume, or switch microphones, repeat the wizard so the thresholds stay accurate.

---

## 🔄 Models & reasoning

* **ASR**: OpenAI Whisper → **Faster-Whisper** (lower latency on CPU).
* **LLM**: Llama (strong bilingual) + tests with **Qwen-2.5 3B** / **Phi-3 Mini 3.8B**.
* **TTS**: **Piper** (fast, local). Fallback: `pyttsx3`.
* **Containerization**: boosts reliability (consistent deps).
* **“Teaser while thinking”**: dropped (complexity > small benefit).

---

## 🗜️ Barge-in reliability (with and without Picovoice)

* **Without Picovoice**: WebRTC VAD + tuned thresholds can pause TTS when **human voice** is detected.
* **With Picovoice**: **Cobra VAD** is more robust to noise; **Porcupine** gives instant wake.
* If you don’t have keys, fallback to text matching for wake and to WebRTC VAD for barge-in.
* The ONNX stop keyword detector watches 1s windows (0.5s hop) while TTS speaks and cuts playback when the “stop robot” logit margin/probability crosses your configured thresholds.

**Pro-tips**
* Raise `min_speech_duration` to avoid coughs/knocks.
* Use voiced-only gating for barge-in.
* Always select **`ec_mic`**.

---

## 🧠 LLM prompt (edit to your goals)

Tweak `configs/llm.yaml` (persona, safety rails, bilingual tone, tools, style, facts mode).

---

## 🛠️ Commands recap

* **Run app with logs**
```bash
LOG_LEVEL=INFO LOG_DIR=logs python -m src.app
```

* **Load AEC** (see Linux audio)
* **Set default mic to `ec_mic`**
* **Verify**: `pactl list short sources | grep -Ei 'ec_mic|echo|cancel'`

---

## 🔜 To-do (next iterations)

* **Instant feedback while thinking** — quick filler if the first token is slow, then keep streaming.
* **Model bake-off** — compare **Phi-3 Mini (3.8B)** vs **Qwen-2.5 (3B)** vs current **Llama** (latency / fluency / bilingual accuracy).

---

## 📸 Vitals & diagram placeholders

![TTS AEC Schema](src/utils/tts_schema.png)

![Robot Vitals](src/utils/vitals.png)
