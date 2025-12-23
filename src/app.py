# src/app.py
from pathlib import Path
from typing import Optional
import os
import time
import threading
import signal
import queue
from rapidfuzz import fuzz
from dotenv import load_dotenv, find_dotenv

# Încarcă variabilele din .env (ex: GROQ_API_KEY)
load_dotenv(find_dotenv())

from src.core.fast_exit import FastExit
from src.core.states import BotState
from src.core.logger import setup_logger
from src.core.config import load_all
from src.audio.input import record_until_silence
from src.audio.barge import BargeInListener
from src.asr import make_asr
from src.llm import make_llm
from src.tts import make_tts
from src.core.wake import WakeDetector
from src.wake.openwakeword_engine import OpenWakeWordEngine
from src.utils.textnorm import normalize_text
from src.audio.openwakeword_listener import OpenWakeWordListener
from src.llm.stream_shaper import shape_stream  # netezire stream LLM→TTS

from src.telemetry.metrics import (
    boot_metrics, round_trip, wake_triggers, sessions_started,
    sessions_ended, interactions, unknown_answer, errors_total,
    tts_speak_calls, log_metrics_snapshot
)

LANG_MAP = {"ro": "ro", "en": "en"}

# 1) încarcă .env din CWD (nu suprascrie ENV deja setate)
load_dotenv(find_dotenv(".env", usecwd=True), override=False)

# 2) root = Conversational_Bot
ROOT = Path(__file__).resolve().parents[1]

# 3) încearcă și repo/.env + configs/.env
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / "configs" / ".env", override=False)


def _lang_from_code(code: str) -> str:
    code = (code or "en").lower()
    for k in LANG_MAP:
        if code.startswith(k):
            return LANG_MAP[k]
    return "en"


def _detect_response_lang(text: str) -> str:
    """Detectează limba răspunsului pe baza caracterelor românești."""
    if not text:
        return "en"
    # Caractere specifice românei
    ro_chars = set("ăâîșțĂÂÎȘȚ")
    ro_count = sum(1 for c in text if c in ro_chars)
    # Dacă are caractere românești, e română
    if ro_count >= 2:
        return "ro"
    # Verifică și cuvinte comune românești
    ro_words = ["este", "pentru", "care", "sunt", "acest", "aceasta", "poate", "doar", "foarte"]
    text_lower = text.lower()
    ro_word_count = sum(1 for w in ro_words if w in text_lower)
    if ro_word_count >= 2:
        return "ro"
    return "en"


def _normalize_phrase(value: str) -> str:
    try:
        return normalize_text(value or "").lower().strip()
    except Exception:
        return ""



def main():
    logger = setup_logger()
    addr, port = boot_metrics()
    logger.info(f"📈 Metrics UI: http://{addr}:{port}/vitals  |  Prometheus: http://{addr}:{port}/metrics")

    cfg = load_all()
    data_dir = Path(cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Engines
    asr = make_asr(cfg["asr"], logger)
    llm = make_llm(cfg["llm"], logger)
    tts = make_tts(cfg["tts"], logger)
    shutdown_once = threading.Event()

    def shutdown_requested() -> bool:
        return shutdown_once.is_set()

    def request_shutdown(reason: str):
        if shutdown_once.is_set():
            return
        shutdown_once.set()
        logger.info(f"🛑 {reason} — opresc TTS, curăț bufferele și raportez metricile.")
        try:
            if tts:
                tts.stop()
        except Exception as exc:
            logger.warning(f"TTS stop error: {exc}")
        try:
            log_metrics_snapshot(logger)
        except Exception:
            pass

    def _handle_sigint(signum, frame):
        request_shutdown("CTRL+C detectat")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    # Wake options
    wake_cfg = cfg.get("wake") or {}
    wake = WakeDetector(wake_cfg, logger)
    ack_cfg = (wake_cfg.get("acknowledgement") or {}) or {}
    ack_en = ack_cfg.get("en") or next(iter(ack_cfg.values()), "Yes, I'm listening.")
    ack_ro = ack_cfg.get("ro", ack_en)

    openwake_cfg = wake_cfg.get("openwakeword") or {}
    wake_keyword = (openwake_cfg.get("wake_keyword") or "hello_robot").strip() or "hello_robot"
    wake_lang = _lang_from_code(openwake_cfg.get("wake_lang") or "en")
    requested_engine = (os.getenv("WAKE_ENGINE") or wake_cfg.get("engine") or "").strip().lower()
    openwake_engine: Optional[OpenWakeWordEngine] = None
    active_engine = "text"

    if requested_engine in ("openwakeword", "openwake"):
        try:
            openwake_engine = OpenWakeWordEngine(cfg["audio"], openwake_cfg, logger)
            if openwake_engine.has_keyword(wake_keyword):
                active_engine = "openwakeword"
            else:
                logger.warning(f"openwakeword: keyword-ul '{wake_keyword}' nu este încărcat — revin pe text wake.")
        except Exception as e:
            logger.warning(f"OpenWakeWord indisponibil: {e} — revin pe wake via text (ASR).")
            openwake_engine = None
    elif requested_engine not in ("", "text"):
        logger.warning(f"Wake engine '{requested_engine}' nu este suportat — folosesc text wake.")

    if active_engine == "openwakeword":
        logger.info("🔔 Wake engine: openwakeword")
        logger.info(f"🎧 Standby (OpenWakeWord) — model pentru „{wake_keyword}” (lang={wake_lang}).")
        logger.info("🤖 Standby: spune „hello robot” ca să pornești conversația.")
    else:
        logger.info("🔔 Wake engine: text")
        logger.info("ℹ️ Wake fallback: recunosc wake phrase-ul din transcript (ASR).")
        logger.info("🤖 Standby: spune „hello robot” ca să pornești conversația.")

    state = BotState.LISTENING
    fast_exit_cfg = (cfg.get("fast_exit") or cfg.get("core", {}).get("fast_exit") or {})
    fast_exit = FastExit(tts, llm, state, logger, fast_exit_cfg, barge=None)
    fast_exit_hotword_cfg = (fast_exit_cfg.get("hotword") or {})
    goodbye_engine = (fast_exit_hotword_cfg.get("engine") or "openwakeword").lower()
    use_fast_exit_hotword = bool(fast_exit_hotword_cfg.get("enabled"))
    fast_exit_listener_cfg: Optional[dict] = None
    goodbye_listener: Optional[OpenWakeWordListener] = None

    if use_fast_exit_hotword:
        if goodbye_engine != "openwakeword":
            logger.warning(f"Goodbye hotword engine '{goodbye_engine}' nesuportat — dezactivez.")
            use_fast_exit_hotword = False
        else:
            bye_path = (
                os.getenv("OPENWAKE_GOODBYE_MODEL", "").strip()
                or str(fast_exit_hotword_cfg.get("model_path") or "").strip()
            )
            if not bye_path:
                logger.warning("🔕 Goodbye hotword (openwakeword): lipsește model_path.")
                use_fast_exit_hotword = False
            elif not Path(bye_path).expanduser().exists():
                logger.warning(f"🔕 Goodbye hotword model lipsă: {bye_path}")
                use_fast_exit_hotword = False
            else:
                fast_exit_listener_cfg = dict(fast_exit_hotword_cfg)
                fast_exit_listener_cfg["model_path"] = str(Path(bye_path).expanduser())
                fast_exit_listener_cfg.setdefault("label", fast_exit_hotword_cfg.get("label") or "goodbye robot")
                fast_exit_listener_cfg.setdefault("threshold", 0.5)
                fast_exit_listener_cfg.setdefault("min_gap_ms", fast_exit_listener_cfg.get("cooldown_ms", 1200))

    if use_fast_exit_hotword and fast_exit_listener_cfg:
        logger.info("🟥 Goodbye hotword disponibil (openwakeword): spune „{label}” ca să închizi sesiunea.".format(
            label=fast_exit_listener_cfg.get("label", "goodbye robot")))
    elif fast_exit_hotword_cfg.get("enabled"):
        logger.info("🟥 Goodbye hotword dezactivat (config incomplet sau eroare).")

    # Încercăm să ne conectăm la "partial" / "final" dacă ASR expune callback-uri.
    try:
        # VARIANTA A: atribut direct on_partial
        old_partial_cb = getattr(asr, "on_partial", None)
        if callable(old_partial_cb) or hasattr(asr, "on_partial"):
            def _combined_partial_cb(text, *a, **kw):
                if fast_exit.on_partial(text):
                    return  # consumă evenimentul -> oprește streamul
                if callable(old_partial_cb):
                    return old_partial_cb(text, *a, **kw)
            asr.on_partial = _combined_partial_cb
        # VARIANTA B: registru de callback-uri
        elif hasattr(asr, "register_callback"):
            try:
                asr.register_callback("partial", lambda text, *a, **kw: fast_exit.on_partial(text))
            except Exception:
                pass
        elif hasattr(asr, "add_listener"):
            try:
                asr.add_listener("partial", lambda text, *a, **kw: fast_exit.on_partial(text))
            except Exception:
                pass

        # Fallback și pe transcriptul final
        if hasattr(asr, "on_final"):
            old_final_cb = getattr(asr, "on_final", None)
            def _combined_final_cb(text, *a, **kw):
                if fast_exit.on_final(text):
                    return
                if callable(old_final_cb):
                    return old_final_cb(text, *a, **kw)
            asr.on_final = _combined_final_cb
    except Exception:
        logger.debug("FastExit: ASR nu expune hook-uri de partial/final — continui fără.")

    last_bot_reply = ""  # anti-eco

    try:
        while not shutdown_requested():
            # —— STANDBY: OpenWakeWord sau fallback text ——
            if active_engine == "openwakeword" and openwake_engine:
                ok = openwake_engine.wait_for(wake_keyword, timeout_seconds=0.25)
                if shutdown_requested():
                    break
                if not ok:
                    time.sleep(0.1)
                    continue
                heard_lang = wake_lang
                wake_triggers.inc()
                logger.info(f"🔔 Wake phrase detectată (openwakeword:{wake_keyword})")
            else:
                # —— STANDBY: text-ASR + fuzzy match ——
                if shutdown_requested():
                    break
                standby_cfg = dict(cfg["audio"])
                standby_cfg.update({
                    "silence_ms_to_end": 1000,
                    "max_record_seconds": 4,
                    "vad_aggressiveness": 3,
                    "min_valid_seconds": 0.7,
                })
                standby_wav = data_dir / "cache" / "standby.wav"
                standby_wav.parent.mkdir(parents=True, exist_ok=True)
                path, dur = record_until_silence(standby_cfg, standby_wav, logger)

                if dur < float(standby_cfg.get("min_valid_seconds", 0.7)):
                    logger.info(f"⏭️ standby prea scurt (dur={dur:.2f}s) — reiau")
                    continue

                # forțăm EN în standby
                result = asr.transcribe(path, language_override="en")
                heard_text = (result.get("text") or "").strip()
                heard_lang = "en"

                scores = wake.debug_scores(heard_text)
                logger.info(f"👂 [standby:{heard_lang}] {heard_text} | wake-scores: {scores}")

                if not heard_text:
                    if shutdown_requested():
                        break
                    continue

                matched = wake.match(heard_text)
                if not matched:
                    if shutdown_requested():
                        break
                    continue

                logger.info(f"🔔 Wake phrase detectată: {matched}")
                wake_triggers.inc()
                matched_norm = normalize_text(matched)
                ro_phrases = [normalize_text(p) for p in cfg["wake"]["wake_phrases"]
                              if "robot" in p and any(x in p.lower() for x in ["salut", "hei", "bun"])]
                heard_lang = "ro" if any(matched_norm == rp for rp in ro_phrases) else "en"

            # —— Wake confirm ——
            ack_key = "ack_ro" if heard_lang == "ro" else "ack_en"
            tts_speak_calls.inc()
            if not tts.say_cached(ack_key, lang=heard_lang):
                ack = ack_ro if heard_lang == "ro" else ack_en
                tts.say(ack, lang=heard_lang)

            # —— SESIUNE MULTI-TURN ——
            ask_cfg = dict(cfg["audio"])
            ask_cfg.update({
                # scurtează endpointing-ul în sesiune (nu afectează standby)
                "silence_ms_to_end": 500,        # echilibru între rapiditate și false positives
                "max_record_seconds": int(cfg["audio"].get("max_record_seconds", 6)),
                "vad_aggressiveness": int(cfg["audio"].get("vad_aggressiveness", 2)),

                # important: permite utterance scurt pentru "goodbye robot"
                "min_valid_seconds": 0.35,       # permiți fraze foarte scurte
            })

            logger.info("🟢 Sesiune activă (spune „goodbye robot” ca să închizi).")
            state = BotState.LISTENING
            sessions_started.inc()

            fast_exit.reset()

            # inițializări lipsă (FIX)
            session_idle_seconds = int(cfg["audio"].get("session_idle_seconds", 12))
            last_activity = time.time()
            goodbye_listener = None
            
            # Conversation history pentru sesiunea curentă
            conversation_history = []

            if use_fast_exit_hotword and fast_exit_listener_cfg:
                def _goodbye_cb(_label: str, *_a):
                    logger.info("🔴 Goodbye hotword detectat — redau mesaj de la revedere.")
                    # Oprește TTS-ul curent dacă vorbește
                    try:
                        tts.stop()
                    except Exception:
                        pass
                    # Redă mesajul de goodbye
                    try:
                        if not tts.say_cached("goodbye_en", lang="en"):
                            tts.say("Goodbye! Have a great day!", lang="en")
                        # Așteaptă să termine de vorbit
                        import time as _time
                        while tts.is_speaking():
                            _time.sleep(0.05)
                    except Exception as e:
                        logger.warning(f"Goodbye TTS error: {e}")
                    # Acum trigger exit
                    fast_exit.trigger_exit("goodbye-hotword")
                try:
                    goodbye_listener = OpenWakeWordListener(
                        cfg_audio=cfg["audio"],
                        cfg_openwake=fast_exit_listener_cfg,
                        logger=logger,
                        on_detect=_goodbye_cb,
                    )
                    goodbye_listener.start()
                    logger.info("🟥 Goodbye hotword activ (openwakeword): spune „goodbye robot” ca să închizi sesiunea.")
                except Exception as e:
                    logger.warning(f"🔕 Goodbye hotword dezactivat pentru sesiunea curentă: {e}")
                    goodbye_listener = None
            short_utt_count = 0  # contor pentru utterance-uri prea scurte
            try:
                while time.time() - last_activity < session_idle_seconds:
                    user_wav = data_dir / "cache" / "user_utt.wav"
                    path_user, dur = record_until_silence(ask_cfg, user_wav, logger, quiet_short=True)

                    if dur < float(ask_cfg.get("min_valid_seconds", 0.35)):
                        short_utt_count += 1
                        continue
                    
                    # Loghează grupat dacă au fost utterance-uri scurte
                    if short_utt_count > 0:
                        logger.info(f"⏭️ Ignorate {short_utt_count} utterance-uri prea scurte")
                        short_utt_count = 0
                    
                    state = BotState.THINKING

                    # ——— ASR: strict RO/EN ———
                    asr_res = None
                    user_text = ""
                    user_lang = "en"
                    try:
                        if hasattr(asr, "transcribe_ro_en"):
                            asr_res = asr.transcribe_ro_en(path_user)
                        else:
                            asr_res = asr.transcribe(path_user, language_override="en")
                        user_text = (asr_res.get("text") or "").strip()
                        user_lang = asr_res.get("lang", "en")
                        if user_lang not in ("ro", "en"):
                            user_lang = "en"
                    except Exception:
                        asr_res = {"text": "", "lang": "en"}
                        user_text = ""
                        user_lang = "en"

                    logger.info(f"🧏 [{user_lang}] {user_text}")

                    # ——— Anti-eco textual ———
                    try:
                        ut = normalize_text(user_text)
                        bt = normalize_text(last_bot_reply)
                        if len(ut) > 8 and len(bt) > 8:
                            sim = fuzz.partial_ratio(ut, bt)
                            if sim >= 85:
                                logger.info(f"🔇 Ignor input (eco TTS) sim={sim}")
                                continue
                    except Exception:
                        pass

                    if not user_text:
                        continue

                    user_text_norm = _normalize_phrase(user_text)
                    # FastExit (inclusiv pe transcript final)
                    if fast_exit.on_final(user_text):
                        logger.info("🔴 FastExit: închis pe transcript final.")
                        break

                    # ——— STREAMING: LLM → TTS ———
                    interactions.inc()
                    rt_start = time.perf_counter()

                    # === Debug dir per sesiune ===
                    from datetime import datetime
                    from src.utils.debug_speech import DebugSpeech
                    session_dir = data_dir / "debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
                    debugger = DebugSpeech(session_dir, user_lang, logger)
                    debugger.write_asr(user_text)

                    reply_buf = []
                    first_token_event = threading.Event()
                    ttft_value = {"value": None}
                    token_queue: "queue.Queue" = queue.Queue()
                    queue_sentinel = object()

                    def _capture(gen):
                        # tee generatorul cu debugger.tee
                        for tok in debugger.tee(gen):
                            reply_buf.append(tok)
                            yield tok

                    # Adaugă mesajul user în history
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    token_iter_raw = llm.generate_stream(user_text, lang_hint=user_lang, mode="precise", history=conversation_history[:-1])

                    # netezește streamul în fraze stabile:
                    tts_cfg = cfg["tts"]
                    min_chunk_chars = int(tts_cfg.get("min_chunk_chars", 60))
                    shaped = shape_stream(
                        token_iter_raw,
                        prebuffer_chars=int(tts_cfg.get("prebuffer_chars", 120)),
                        min_chunk_chars=min_chunk_chars,
                        soft_max_chars=int(tts_cfg.get("soft_max_chars", 140)),
                        max_idle_ms=int(tts_cfg.get("max_idle_ms", 250)),
                    )

                    # Capture + gard de oprire
                    def _abort_guard(gen):
                        for tok in gen:
                            if fast_exit.pending():
                                break
                            yield tok

                    def _producer():
                        try:
                            first_local = True
                            for tok in _capture(_abort_guard(shaped)):
                                if first_local:
                                    first_local = False
                                    ttft_value["value"] = time.perf_counter() - rt_start
                                    first_token_event.set()
                                token_queue.put(tok)
                        finally:
                            token_queue.put(queue_sentinel)

                    producer_th = threading.Thread(target=_producer, name="LLMTokenProducer", daemon=True)
                    producer_th.start()

                    def _queue_iter():
                        while True:
                            item = token_queue.get()
                            if item is queue_sentinel:
                                break
                            yield item

                    backchannel_cfg = tts_cfg.get("backchannel") or {}
                    backchannel_enabled = bool(backchannel_cfg.get("enabled", True))
                    backchannel_delay = float(backchannel_cfg.get("delay_ms", 2000)) / 1000.0
                    backchannel_phrase_en = backchannel_cfg.get("phrase_en") or "One moment..."
                    backchannel_phrase_ro = backchannel_cfg.get("phrase_ro") or "Un moment..."
                    if backchannel_enabled and backchannel_delay > 0.0:
                        if not first_token_event.wait(backchannel_delay) and not fast_exit.pending():
                            filler_key = "filler_ro" if user_lang.startswith("ro") else "filler_en"
                            logger.info("⌛ Backchannel: TTFT depășește %.1fs — redau filler.", backchannel_delay)
                            try:
                                if not tts.say_cached(filler_key, lang=user_lang):
                                    phrase = backchannel_phrase_ro if user_lang.startswith("ro") else backchannel_phrase_en
                                    tts.say(phrase, lang=user_lang)
                            except Exception as exc:
                                logger.warning(f"Backchannel TTS error: {exc}")
                    token_iter = _queue_iter()
                    if fast_exit.pending():
                        logger.info("🔴 FastExit activ înainte de TTS — abandonez răspunsul curent.")
                        break

                    def _mark_tts_start():
                        # round-trip metric
                        round_trip.observe(time.perf_counter() - rt_start)
                        # debug hook
                        debugger.on_tts_start()

                    # Folosim direct limba userului pentru TTS (nu detectăm din răspuns)
                    # Așa TTS va vorbi în română când userul întreabă în română
                    response_lang = user_lang
                    logger.info(f"🌐 TTS va folosi limba input-ului: {response_lang}")
                    
                    final_token_iter = token_iter

                    state = BotState.SPEAKING
                    tts_speak_calls.inc()
                    tts.say_async_stream(
                        final_token_iter,
                        lang=response_lang,
                        on_first_speak=_mark_tts_start,
                        min_chunk_chars=min_chunk_chars,
                    )


                    # BARGE-IN în timpul TTS (protejată anti-eco și cu arm-delay)
                    # Stop keyword detector rulează ÎNTOTDEAUNA, barge-in pe voce e opțional
                    barge = BargeInListener(cfg["audio"], logger)
                    fast_exit.barge = barge
                    barge_on_voice = bool(cfg["audio"].get("barge_enabled", False)) and bool(cfg["audio"].get("barge_allow_during_tts", True))
                    try:
                        while tts.is_speaking():
                            if fast_exit.pending():
                                tts.stop()
                                break
                            # heard_speech verifică intern stop_detector și returnează True dacă a detectat "stop"
                            need = int(cfg["audio"].get("barge_min_voice_ms", 650))
                            detected = barge.heard_speech(need_ms=need)
                            if detected:
                                # Stop keyword sau voce detectată
                                if barge_on_voice:
                                    logger.info("⛔ Barge-in detectat — opresc TTS și trec la listening.")
                                tts.stop()
                                break
                            time.sleep(0.03)
                    finally:
                        barge.close()

                    # finalizează logurile
                    debugger.on_tts_end()
                    last_bot_reply = "".join(reply_buf)
                    
                    # Adaugă răspunsul bot în history
                    if last_bot_reply.strip():
                        conversation_history.append({"role": "assistant", "content": last_bot_reply})
                    
                    debugger.finish()
                    if fast_exit.pending():
                        logger.info("🔴 FastExit: sesiune închisă (revenire în standby).")
                        break

                    last_activity = time.time()
            finally:
                if goodbye_listener:
                    goodbye_listener.stop()

            # —— ieșire din sesiune => standby ——
            state = BotState.LISTENING
            if shutdown_requested():
                break
            logger.info("⏳ Revenire în standby (spune din nou wake-phrase pentru o nouă sesiune).")
            sessions_ended.inc()

    except KeyboardInterrupt:
        request_shutdown("CTRL+C detectat")
        logger.info("Bye!")
    except Exception as e:
        errors_total.inc()
        request_shutdown("Eroare fatală")
        logger.exception(f"Fatal error: {e}")
    finally:
        request_shutdown("Închid aplicația")
        try:
            wake.close()
        except Exception:
            pass
        try:
            if openwake_engine:
                openwake_engine.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
