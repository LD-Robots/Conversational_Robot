# src/server/api.py
"""
Server API REST pentru procesare remote ASR/LLM/TTS.

Rulează pe laptop-ul "server" (cel cu putere de procesare).
Clientul trimite audio/text și primește înapoi text/audio.

Usage:
    python -m src.server.api --host 0.0.0.0 --port 8001
    
Sau pentru test local:
    python -m src.server.api --host 127.0.0.1 --port 8001
"""
from __future__ import annotations
import os
import sys
import tempfile
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, Response, send_file
from dotenv import load_dotenv, find_dotenv

# Încarcă .env pentru GROQ_API_KEY etc.
load_dotenv(find_dotenv())

# Adaugă root-ul proiectului în path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.config import load_all
from src.core.logger import setup_logger

app = Flask(__name__)

# Global instances - inițializate la startup
_asr = None
_llm = None
_tts_cfg = None
_logger = None


def _init_engines():
    """Inițializează engine-urile la pornirea serverului."""
    global _asr, _llm, _tts_cfg, _logger
    
    _logger = setup_logger("server")
    _logger.info("🚀 Inițializez engine-urile pentru server...")
    
    cfg = load_all()
    
    # ASR - folosim direct engine-ul, nu factory-ul (care ar putea returna Remote)
    from src.asr.engine_faster import ASREngine
    _asr = ASREngine(
        model_size=cfg["asr"].get("model_size", "small"),
        compute_type=cfg["asr"].get("compute_type", "int8"),
        device=cfg["asr"].get("device", "cpu"),
        force_language=cfg["asr"].get("force_language"),
        beam_size=int(cfg["asr"].get("beam_size", 1)),
        vad_min_silence_ms=int(cfg["asr"].get("vad_min_silence_ms", 300)),
        warmup_enabled=bool(cfg["asr"].get("warmup_enabled", True)),
        logger=_logger,
    )
    
    # LLM - folosim direct engine-ul
    from src.llm.engine import LLMLocal
    _llm = LLMLocal(cfg["llm"], _logger)
    
    # TTS config - pentru sinteză
    _tts_cfg = cfg["tts"]
    
    _logger.info("✅ Server gata! Aștept cereri...")


# ─────────────────────────────────────────────────────────────
# ASR Endpoints
# ─────────────────────────────────────────────────────────────

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcrie audio WAV în text.
    
    Request:
        Body: raw audio WAV bytes
        Query params: language (optional) - forțează o limbă
        
    Response:
        JSON: {"text": "...", "lang": "en/ro", "language_probability": 0.95}
    """
    try:
        audio_data = request.data
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400
        
        language = request.args.get('language')
        
        # Salvează audio în fișier temporar
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="server_asr_")
        os.close(fd)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            result = _asr.transcribe(temp_path, language_override=language)
            _logger.info(f"🧏 ASR: [{result.get('lang')}] {result.get('text', '')[:50]}...")
            
            return jsonify(result)
            
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
                
    except Exception as e:
        _logger.error(f"ASR error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe_ro_en', methods=['POST'])
def transcribe_ro_en():
    """
    Transcrie audio cu detecție automată RO/EN.
    Rulează transcriere în ambele limbi și alege cea mai bună.
    
    Request:
        Body: raw audio WAV bytes
        
    Response:
        JSON: {"text": "...", "lang": "en/ro", "language_probability": 1.0}
    """
    try:
        audio_data = request.data
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400
        
        # Salvează audio în fișier temporar
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="server_asr_")
        os.close(fd)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            result = _asr.transcribe_ro_en(temp_path)
            _logger.info(f"🧏 ASR (ro_en): [{result.get('lang')}] {result.get('text', '')[:50]}...")
            
            return jsonify(result)
            
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
                
    except Exception as e:
        _logger.error(f"ASR error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# LLM Endpoints
# ─────────────────────────────────────────────────────────────

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generează răspuns LLM (non-streaming).
    
    Request:
        JSON: {"text": "user message", "lang": "en/ro", "mode": "precise"}
        
    Response:
        JSON: {"response": "..."}
    """
    try:
        data = request.json or {}
        user_text = data.get("text", "")
        lang = data.get("lang", "en")
        mode = data.get("mode")
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        response = _llm.generate(user_text, lang_hint=lang, mode=mode)
        _logger.info(f"🧠 LLM: {response[:50]}...")
        
        return jsonify({"response": response})
        
    except Exception as e:
        _logger.error(f"LLM error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    """
    Generează răspuns LLM cu streaming.
    
    Request:
        JSON: {"text": "user message", "lang": "en/ro", "mode": "precise", "history": [...]}
        
    Response:
        Stream de tokens (text/plain), fiecare token pe o linie nouă
    """
    try:
        data = request.json or {}
        user_text = data.get("text", "")
        lang = data.get("lang", "en")
        mode = data.get("mode")
        history = data.get("history", [])
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        def generate_tokens():
            try:
                for token in _llm.generate_stream(user_text, lang_hint=lang, mode=mode, history=history):
                    yield token + "\n"
            except Exception as e:
                _logger.error(f"LLM stream error: {e}")
        
        _logger.info(f"🧠 LLM stream start: {user_text[:30]}...")
        return Response(generate_tokens(), mimetype='text/plain')
        
    except Exception as e:
        _logger.error(f"LLM stream error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# TTS Endpoints
# ─────────────────────────────────────────────────────────────

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Sintetizează text în audio MP3.
    
    Request:
        JSON: {"text": "text to speak", "lang": "en/ro"}
        
    Response:
        Audio MP3 binary
    """
    try:
        data = request.json or {}
        text = data.get("text", "")
        lang = data.get("lang", "en")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Folosim Edge TTS pentru sinteză
        import edge_tts
        
        voice_en = _tts_cfg.get("edge_voice_en", "en-GB-SoniaNeural")
        voice_ro = _tts_cfg.get("edge_voice_ro", "ro-RO-EmilNeural")
        voice = voice_ro if lang.lower().startswith("ro") else voice_en
        
        rate = _tts_cfg.get("edge_rate", "+0%")
        pitch = _tts_cfg.get("edge_pitch", "+0Hz")
        
        # Generează audio
        fd, temp_path = tempfile.mkstemp(suffix=".mp3", prefix="server_tts_")
        os.close(fd)
        
        try:
            async def synth():
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await communicate.save(temp_path)
            
            asyncio.run(synth())
            
            _logger.info(f"🗣️ TTS: [{lang}] {text[:30]}...")
            
            # Trimite fișierul audio
            return send_file(
                temp_path,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name='response.mp3'
            )
            
        finally:
            # Cleanup se face după ce flask trimite fișierul
            # (în producție ar trebui un mecanism mai bun)
            pass
                
    except Exception as e:
        _logger.error(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Verifică că serverul funcționează."""
    return jsonify({
        "status": "ok",
        "asr": _asr is not None,
        "llm": _llm is not None,
    })


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Server API pentru ASR/LLM/TTS")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Inițializează engine-urile
    _init_engines()
    
    print(f"\n🌐 Server pornit: http://{args.host}:{args.port}")
    print(f"   Health check:  http://{args.host}:{args.port}/health")
    print(f"   Endpoints:")
    print(f"     POST /transcribe      - ASR (audio → text)")
    print(f"     POST /transcribe_ro_en - ASR bilingv")
    print(f"     POST /generate        - LLM (text → text)")
    print(f"     POST /generate_stream - LLM streaming")
    print(f"     POST /synthesize      - TTS (text → audio)")
    print(f"\n   Apasă Ctrl+C pentru a opri.\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
