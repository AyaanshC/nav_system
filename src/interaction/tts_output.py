import pyttsx3
import threading
import queue
from loguru import logger
import time

class TTSOutput:
    """
    Multimodal User Interaction: Voice Output.
    Uses pyttsx3 for low-latency audio guidance.
    Implements Safety Preemption for immediate obstacle alerts.
    """
    def __init__(self):
        self._queue = queue.Queue()
        self._stop_event = threading.Event()

        # We need a separate pyttsx3 engine instance per thread or init it in the thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # For immediate preemption
        self._preempt_engine = pyttsx3.init()
        self._preempt_engine.setProperty('rate', 160)

        logger.info("[TTS] Output system initialized.")

    def _worker(self):
        """Background thread processing normal TTS messages."""
        # Initialize engine inside thread
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        while True:
            try:
                # Block until a message is available
                message = self._queue.get()

                # If a stop was requested while we were waiting, clear and continue
                if self._stop_event.is_set():
                    self._stop_event.clear()
                    self._queue.task_done()
                    continue

                engine.say(message)

                # We need a way to break out of runAndWait if stop is set,
                # but pyttsx3's runAndWait is blocking.
                # To simulate preemption, we will use a separate engine for alerts.
                engine.runAndWait()

                self._queue.task_done()

            except Exception as e:
                logger.error(f"[TTS] Worker error: {e}")

    def speak(self, text: str):
        """Queues a normal message to be spoken."""
        logger.debug(f"[TTS] Queueing: {text}")
        self._queue.put(text)

    def speak_alert(self, text: str):
        """
        Immediate Safety Preemption.
        Clears the current queue and speaks instantly using a separate engine
        (since python's pyttsx3 main loop might be blocked, this isn't perfect
        multi-threading, but in OS level it often allows overlapping audio or
        interrupts depending on the backend).
        """
        logger.warning(f"[TTS] PREEMPTIVE ALERT: {text}")

        # Signal worker to drop pending items
        self._stop_event.set()

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Play alert instantly on main thread
        try:
            self._preempt_engine.say(text)
            self._preempt_engine.runAndWait()
        except Exception as e:
            logger.error(f"[TTS] Preempt error: {e}")

    def shutdown(self):
        # Allow queue to drain or forcefully exit
        pass
