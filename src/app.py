# src/app.py
from pathlib import Path
import os
import time
import threading
from rapidfuzz import fuzz
from dotenv import load_dotenv, find_dotenv

from src.core.fast_exit import FastExit
from src.core.states import BotState
from src.core.logger import setup_logger
from src.core.config import load_all
from src.audio.input import record_until_silence
from src.audio.barge import BargeInListener
from src.asr import make_asr
from src.llm.engine import LLMLocal
from src.tts.engine import TTSLocal
from src.core.wake import WakeDetector
from src.utils.textnorm import normalize_text
from src.audio.wake_porcupine import wait_for_wake as wait_for_wake_porcupine
from src.audio.picovoice_stop import PicovoiceStopListener
from src.llm.stream_shaper import shape_stream  # netezire stream LLM→TTS

from src.telemetry.metrics import (
    boot_metrics, round_trip, wake_triggers, sessions_started,
    sessions_ended, interactions, unknown_answer, errors_total,
    tts_speak_calls
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


def is_goodbye(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    # doar potriviri exacte; evităm trigger la cuvinte mai lungi (ex: "paine")
    bye_exact = {"goodbye robot", "good bye robot"}
    return t in bye_exact


def main():
    logger = setup_logger()
    addr, port = boot_metrics()
    logger.info(f"📈 Metrics UI: http://{addr}:{port}/vitals  |  Prometheus: http://{addr}:{port}/metrics")

    cfg = load_all()
    data_dir = Path(cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Engines
    asr = make_asr(cfg["asr"], logger)
    llm = LLMLocal(cfg["llm"], logger)
    tts = TTSLocal(cfg["tts"], logger)

    # Wake options
    wake = WakeDetector(cfg["wake"], logger)
    ack_ro = cfg["wake"]["acknowledgement"]["ro"]
    ack_en = cfg["wake"]["acknowledgement"]["en"]

    # -------- Wake config (porcupine) --------
    PV_KEY = (
        os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
        or (cfg["wake"].get("porcupine", {}) or {}).get("access_key", "").strip()
    )
    PPN_PATH = (
        os.getenv("PORCUPINE_PPN", "").strip()
        or next(iter((cfg["wake"].get("porcupine", {}) or {}).get("keyword_paths", []) or []), "")
    )
    PORC_SENS = float(os.getenv("PORCUPINE_SENSITIVITY", "0.6"))
    PORC_LANG = (os.getenv("PORCUPINE_LANG", "en") or "en").lower()

    def _validate_porcupine():
        errors = []
        if not PV_KEY:
            errors.append("PICOVOICE_ACCESS_KEY lipsește")
        if not PPN_PATH:
            errors.append("PORCUPINE_PPN lipsește")
        elif not Path(PPN_PATH).exists():
            errors.append(f"ppn missing: {PPN_PATH}")
        return (len(errors) == 0), errors

    use_porcupine = False
    ok, errs = _validate_porcupine()
    if ok:
        use_porcupine = True
    else:
        logger.warning(f"🔕 Porcupine cerut, dar lipsesc: {', '.join(errs)} — revin pe wake via text.")

    active_engine = "porcupine" if use_porcupine else "text"
    logger.info(f"🔔 Wake engine: {active_engine}")
    if not use_porcupine:
        logger.info("ℹ️ Wake fallback: recunosc wake phrase-ul din transcript (ASR).")

    # Circuit-breaker doar pentru Porcupine
    porcupine_failures = 0
    PORCUPINE_MAX_FAILS = 3

    logger.info("🤖 Standby: spune „hello robot” ca să pornești conversația.")
    state = BotState.LISTENING
    fast_exit_cfg = (cfg.get("fast_exit") or cfg.get("core", {}).get("fast_exit") or {})
    fast_exit = FastExit(tts, llm, state, logger, fast_exit_cfg, barge=None)
    stop_barge_event = threading.Event()
    stop_hotword_cfg = (
        cfg.get("stop_hotword")
        or cfg.get("core", {}).get("stop_hotword")
        or fast_exit_cfg.get("picovoice")
        or {}
    )
    stop_mode = (stop_hotword_cfg.get("mode") or "exit").lower()
    stop_label = (stop_hotword_cfg.get("label") or "stop").strip() or "stop"
    stop_access_key = ""
    stop_ppn_path = ""
    stop_sensitivity = 0.6
    use_stop_hotword = bool(stop_hotword_cfg.get("enabled"))
    stop_listener = None

    if use_stop_hotword:
        stop_access_key = (
            (stop_hotword_cfg.get("access_key") or "").strip()
            or PV_KEY
            or os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
        )
        stop_ppn_path = (
            os.getenv("PORCUPINE_STOP_PPN", "").strip()
            or (stop_hotword_cfg.get("keyword_path") or "").strip()
        )
        stop_sens_raw = os.getenv("PORCUPINE_STOP_SENSITIVITY", None)
        if stop_sens_raw in (None, ""):
            stop_sens_raw = stop_hotword_cfg.get("sensitivity", 0.6)
        try:
            stop_sensitivity = float(stop_sens_raw)
        except (TypeError, ValueError):
            stop_sensitivity = 0.6
        if not stop_access_key or not stop_ppn_path:
            logger.warning("🔕 Stop hotword: lipsesc access_key/keyword_path pentru Porcupine.")
            use_stop_hotword = False
        elif not Path(stop_ppn_path).exists():
            logger.warning(f"🔕 Picovoice stop keyword lipsește: {stop_ppn_path}")
            use_stop_hotword = False

    if use_stop_hotword:
        if stop_mode == "barge":
            logger.info(f"🛑 Stop hotword activ (porcupine): spune „{stop_label}” ca să oprești TTS-ul și să revină la ascultare.")
        else:
            logger.info(f"🛑 Stop hotword activ (porcupine): spune „{stop_label}” ca să închizi instant sesiunea.")

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
        while True:
            # —— STANDBY: Porcupine -> text fallback ——
            if use_porcupine:
                ok = wait_for_wake_porcupine(
                    cfg_audio=cfg["audio"],
                    access_key=PV_KEY,
                    keyword_path=PPN_PATH,
                    sensitivity=PORC_SENS,
                    logger=logger,
                    timeout_seconds=None
                )
                if not ok:
                    porcupine_failures += 1
                    if porcupine_failures >= PORCUPINE_MAX_FAILS:
                        logger.warning("⚠️ Porcupine a eșuat repetat — comut pe wake via text pentru sesiunea curentă.")
                        use_porcupine = False
                    time.sleep(0.1)
                    continue
                porcupine_failures = 0
                matched = "wake-porcupine"
                heard_lang = "ro" if PORC_LANG.startswith("ro") else "en"
                logger.info("🔔 Wake phrase detectată (porcupine)")
                wake_triggers.inc()
            else:
                # —— STANDBY: text-ASR + fuzzy match ——
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
                    continue

                matched = wake.match(heard_text)
                if not matched:
                    continue

                logger.info(f"🔔 Wake phrase detectată: {matched}")
                wake_triggers.inc()
                matched_norm = normalize_text(matched)
                ro_phrases = [normalize_text(p) for p in cfg["wake"]["wake_phrases"]
                              if "robot" in p and any(x in p.lower() for x in ["salut", "hei", "bun"])]
                heard_lang = "ro" if any(matched_norm == rp for rp in ro_phrases) else "en"

            # —— Wake confirm ——
            ack = ack_ro if heard_lang == "ro" else ack_en
            tts_speak_calls.inc()
            tts.say(ack, lang=heard_lang)

            # —— SESIUNE MULTI-TURN ——
            ask_cfg = dict(cfg["audio"])
            ask_cfg.update({
                # scurtează endpointing-ul în sesiune (nu afectează standby)
                "silence_ms_to_end": 450,        # de la 1400 -> ~450ms
                "max_record_seconds": int(cfg["audio"].get("max_record_seconds", 6)),
                "vad_aggressiveness": int(cfg["audio"].get("vad_aggressiveness", 3)),

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
            stop_listener = None
            stop_barge_event.clear()
            if use_stop_hotword:
                def _stop_cb(_label: str):
                    if stop_mode == "barge":
                        if tts.is_speaking():
                            logger.info("🟠 Stop hotword detectat — opresc TTS și revin la listening.")
                            stop_barge_event.set()
                        else:
                            logger.info("🟠 Stop hotword detectat, dar TTS nu vorbește — ignor.")
                    else:
                        fast_exit.trigger_exit("stop-hotword")
                try:
                    stop_listener = PicovoiceStopListener(
                        cfg_audio=cfg["audio"],
                        access_key=stop_access_key,
                        keyword_path=stop_ppn_path,
                        sensitivity=stop_sensitivity,
                        label=stop_label,
                        logger=logger,
                        on_detect=_stop_cb,
                    )
                    stop_listener.start()
                except Exception as e:
                    logger.warning(f"🔕 Stop hotword dezactivat pentru sesiunea curentă: {e}")
                    stop_listener = None

            try:
                while time.time() - last_activity < session_idle_seconds:
                    user_wav = data_dir / "cache" / "user_utt.wav"
                    path_user, dur = record_until_silence(ask_cfg, user_wav, logger)

                    if dur < float(ask_cfg.get("min_valid_seconds", 0.35)):
                        continue

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

                    # FastExit (inclusiv pe transcript final)
                    if fast_exit.on_final(user_text):
                        logger.info("🔴 FastExit: închis pe transcript final.")
                        break

                    # închidere sesiune pe "goodbye robot"
                    if is_goodbye(user_text):
                        state = BotState.SPEAKING
                        tts_speak_calls.inc()
                        tts.say("La revedere!" if user_lang == "ro" else "Goodbye!", lang=user_lang)
                        logger.info("🔴 Sesiune închisă de utilizator (goodbye robot).")
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

                    def _capture(gen):
                        # tee generatorul cu debugger.tee
                        for tok in debugger.tee(gen):
                            reply_buf.append(tok)
                            yield tok

                    token_iter_raw = llm.generate_stream(user_text, lang_hint=user_lang, mode="precise")

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

                    token_iter = _capture(_abort_guard(shaped))

                    def _mark_tts_start():
                        # round-trip metric
                        round_trip.observe(time.perf_counter() - rt_start)
                        # debug hook
                        debugger.on_tts_start()

                    state = BotState.SPEAKING
                    tts_speak_calls.inc()
                    stop_barge_event.clear()
                    tts.say_async_stream(
                        token_iter,
                        lang=user_lang,
                        on_first_speak=_mark_tts_start,
                        min_chunk_chars=min_chunk_chars,
                    )

                    # BARGE-IN în timpul TTS (protejată anti-eco și cu arm-delay)
                    if not bool(cfg["audio"].get("barge_enabled", True)):
                        while tts.is_speaking():
                            if fast_exit.pending() or stop_barge_event.is_set():
                                tts.stop()
                                stop_barge_event.clear()
                                break
                            time.sleep(0.05)
                    elif not bool(cfg["audio"].get("barge_allow_during_tts", True)):
                        while tts.is_speaking():
                            if fast_exit.pending() or stop_barge_event.is_set():
                                tts.stop()
                                stop_barge_event.clear()
                                break
                            time.sleep(0.05)
                    else:
                        barge = BargeInListener(cfg["audio"], logger)
                        fast_exit.barge = barge  # permite FastExit să verifice că vorbește userul, nu eco TTS
                        try:
                            while tts.is_speaking():
                                if fast_exit.pending() or stop_barge_event.is_set():
                                    tts.stop()
                                    stop_barge_event.clear()
                                    break
                                need = int(cfg["audio"].get("barge_min_voice_ms", 650))
                                if barge.heard_speech(need_ms=need):
                                    logger.info("⛔ Barge-in detectat — opresc TTS și trec la listening.")
                                    tts.stop()
                                    break
                                time.sleep(0.03)
                        finally:
                            barge.close()

                    # finalizează logurile
                    debugger.on_tts_end()
                    last_bot_reply = "".join(reply_buf)
                    debugger.finish()
                    if fast_exit.pending():
                        logger.info("🔴 FastExit: sesiune închisă (revenire în standby).")
                        break

                    last_activity = time.time()
            finally:
                if stop_listener:
                    stop_listener.stop()

            # —— ieșire din sesiune => standby ——
            state = BotState.LISTENING
            logger.info("⏳ Revenire în standby (spune din nou wake-phrase pentru o nouă sesiune).")
            sessions_ended.inc()

    except KeyboardInterrupt:
        logger.info("Bye!")
    except Exception as e:
        errors_total.inc()
        logger.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
