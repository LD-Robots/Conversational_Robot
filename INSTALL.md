# Installation

> **Toate comenzile se rulează din terminalul VSCode** (Ctrl+`)

## 1. Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install Ollama + Model
```bash
sudo snap install ollama
ollama pull qwen2.5:3b
```

## 3. Install Piper TTS
```bash
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
sudo mv piper /usr/local/bin/
```

## 4. Install System Audio Tools
```bash
sudo apt install pavucontrol pulseaudio portaudio19-dev
```

## 5. Run
```bash
source .venv/bin/activate
LOG_LEVEL=INFO python -m src.app
```
