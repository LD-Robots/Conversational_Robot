from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .devices import choose_input_device


class PicovoiceStopListener:
    """
    Lightweight Porcupine listener that runs in a background thread and fires
    a callback as soon as the configured keyword is detected.
    """

    def __init__(
        self,
        cfg_audio: dict,
        access_key: str,
        keyword_path: str,
        sensitivity: float = 0.6,
        label: str = "stop",
        logger=None,
        on_detect: Optional[Callable[[str], None]] = None,
    ):
        self.cfg_audio = cfg_audio or {}
        self.access_key = (access_key or "").strip()
        self.keyword_path = (keyword_path or "").strip()
        self.sensitivity = float(sensitivity or 0.6)
        self.label = label or "stop"
        self.logger = logger
        self.on_detect = on_detect

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._detected = threading.Event()

        if not self.access_key:
            raise ValueError("PicovoiceStopListener: access_key lipsă.")
        if not self.keyword_path:
            raise ValueError("PicovoiceStopListener: keyword_path lipsă.")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._detected.clear()
        self._thread = threading.Thread(target=self._run, name="PicovoiceStop", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    def detected(self) -> bool:
        return self._detected.is_set()

    # ---- intern ----
    def _run(self):
        try:
            import pvporcupine as pv
        except Exception as e:
            if self.logger:
                self.logger.error(f"Picovoice stop: nu pot importa Porcupine ({e})")
            return

        porcupine = None
        stream = None
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

        try:
            porcupine = pv.create(
                access_key=self.access_key,
                keyword_paths=[self.keyword_path],
                sensitivities=[self.sensitivity],
            )
            sr = porcupine.sample_rate
            frame_len = porcupine.frame_length

            dev_index = choose_input_device(
                prefer_echo_cancel=bool(self.cfg_audio.get("prefer_echo_cancel", True)),
                hint=str(self.cfg_audio.get("input_device_hint", "") or ""),
                logger=self.logger,
            )

            if self.logger:
                self.logger.debug(
                    f"🎧 Picovoice stop listener on device={dev_index} sr={sr} frame={frame_len} sens={self.sensitivity}"
                )

            def callback(indata, frames, time_info, status):
                if status and self.logger:
                    self.logger.debug(f"Picovoice stop input status: {status}")
                try:
                    q.put_nowait(indata.copy())
                except queue.Full:
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                    try:
                        q.put_nowait(indata.copy())
                    except Exception:
                        pass

            stream = sd.InputStream(
                channels=1,
                samplerate=sr,
                blocksize=frame_len,
                dtype="int16",
                callback=callback,
                device=dev_index,
            )
            stream.start()

            while not self._stop.is_set():
                try:
                    block = q.get(timeout=0.25)
                except queue.Empty:
                    continue

                if block.ndim == 2:
                    pcm = block[:, 0]
                else:
                    pcm = block

                if len(pcm) != frame_len:
                    pcm = np.resize(pcm, frame_len).astype("int16")

                res = porcupine.process(pcm)
                if res >= 0:
                    self._detected.set()
                    if self.logger:
                        self.logger.info(f"🔴 Picovoice stop hotword detectat: {self.label}")
                    if callable(self.on_detect):
                        try:
                            self.on_detect(self.label)
                        except Exception as cb_err:
                            if self.logger:
                                self.logger.warning(f"Picovoice stop callback error: {cb_err}")
                    break

        except Exception as e:
            if self.logger:
                self.logger.error(f"Picovoice stop runtime error: {e}")
        finally:
            self._stop.set()
            try:
                if stream:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            try:
                if porcupine:
                    porcupine.delete()
            except Exception:
                pass
