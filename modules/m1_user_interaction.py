"""
modules/m1_user_interaction.py

Two public functions:
  listen_for_goal() -> str | None   — blocking, returns spoken text or None on timeout
  speak(text: str) -> None          — async TTS, NON-BLOCKING (runs in background thread)

Also:
  listen_for_interrupt(callback)    — background listener for voice commands during navigation
"""

import time
import os
import tempfile
import asyncio
import threading
import queue as _queue
import pygame
import speech_recognition as sr
import edge_tts
from loguru import logger
from pathlib import Path


# ── TTS setup ─────────────────────────────────────────────────────────────────

VOICE = "en-US-JennyNeural"           # Natural, clear voice
SPEECH_RATE = "+0%"                    # Adjustable via web UI: "+10%" faster, "-10%" slower
_pygame_initialized = False

# Background TTS worker — speak() puts jobs here, worker thread consumes them
_tts_queue: _queue.Queue = _queue.Queue(maxsize=2)
_tts_worker_started = False


def _init_pygame():
    global _pygame_initialized
    if not _pygame_initialized:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        _pygame_initialized = True


async def _tts_async(text: str, rate: str = SPEECH_RATE) -> str:
    """Generate TTS audio to a temp file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    communicate = edge_tts.Communicate(text, VOICE, rate=rate)
    await communicate.save(tmp.name)
    return tmp.name


def _tts_worker():
    """
    Background daemon thread. Consumes (text, rate) jobs from _tts_queue,
    generates audio via edge-tts, and plays it via pygame.
    Non-blocking from the caller's perspective — sensing pipeline keeps running.
    """
    _init_pygame()
    while True:
        try:
            text, rate = _tts_queue.get(timeout=1)
        except _queue.Empty:
            continue

        try:
            # Generate audio file
            loop = asyncio.new_event_loop()
            path = loop.run_until_complete(_tts_async(text, rate))
            loop.close()

            # Stop any currently playing audio (interrupt old instruction)
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.unload()
                except AttributeError:
                    pass

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                # If a new item arrives while playing, stop early
                if not _tts_queue.empty():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)

            try:
                pygame.mixer.music.unload()
            except AttributeError:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass

        except Exception as e:
            logger.warning(f"[TTS] Worker error: {e}")


def _ensure_worker():
    global _tts_worker_started
    if not _tts_worker_started:
        t = threading.Thread(target=_tts_worker, daemon=True, name="TTS-Worker")
        t.start()
        _tts_worker_started = True


def speak(text: str, rate: str | None = None) -> None:
    """
    Convert text to speech and play it — FULLY NON-BLOCKING.
    Audio plays in a background thread so YOLO/MiDaS/SLAM continue uninterrupted.
    If a new instruction arrives while audio is playing, the old audio is cut short.

    Args:
        text: The text to speak aloud.
        rate: Edge-TTS rate string e.g. "+10%" or "-5%".
              If None, reads the live value from the web UI state.
    """
    if not text:
        return
    if rate is None:
        try:
            from modules.m6_usability import get_tts_rate
            rate = get_tts_rate()
        except Exception:
            rate = SPEECH_RATE
    logger.info(f"[TTS] Speaking: {text[:80]}{'...' if len(text) > 80 else ''}")
    _ensure_worker()

    # Drop oldest item if queue is full (always prioritize newest instruction)
    if _tts_queue.full():
        try:
            _tts_queue.get_nowait()
        except _queue.Empty:
            pass

    _tts_queue.put_nowait((text, rate))


# ── Speech recognition ────────────────────────────────────────────────────────

# Instantiate globally to prevent PyAudio C-level crashes on Windows
# (creating multiple PyAudio instances in different threads is unstable)
_recognizer = sr.Recognizer()
_recognizer.energy_threshold = 300
_recognizer.dynamic_energy_threshold = True

try:
    _mic = sr.Microphone()
except Exception as e:
    logger.warning(f"[STT] Could not initialize microphone: {e}")
    _mic = None

# Global lock to prevent concurrent microphone access
_mic_lock = threading.Lock()

def listen_for_goal(
    timeout: int = 8,
    phrase_time_limit: int = 6,
    max_retries: int = 3
) -> str | None:
    """
    Listen for a navigation goal spoken by the user.
    Retries up to max_retries times before giving up.

    Returns:
        str: transcribed text (e.g. "take me to the restroom")
        None: could not understand after all retries
    """
    if _mic is None:
        logger.error("[STT] Microphone not available.")
        return None

    for attempt in range(1, max_retries + 1):
        logger.info(f"[STT] Listening attempt {attempt}/{max_retries}")
        if attempt == 1:
            speak("Where would you like to go?")
        else:
            speak("Sorry, I didn't catch that. Please say your destination again.")

        # Wait for TTS to finish before listening (so mic doesn't pick up TTS audio)
        time.sleep(0.5)
        while not _tts_queue.empty() or pygame.mixer.music.get_busy():
            time.sleep(0.1)

        with _mic_lock:
            with _mic as source:
                _recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    logger.warning("[STT] No speech detected within timeout.")
                    continue

        try:
            text = _recognizer.recognize_google(audio, language="en-US")
            logger.info(f"[STT] Recognized: '{text}'")
            return text.lower().strip()
        except sr.UnknownValueError:
            logger.warning("[STT] Could not understand audio.")
        except sr.RequestError as e:
            logger.error(f"[STT] Google API error: {e}")
            return None

    speak("I was unable to understand you. Please try again later.")
    return None


# ── Continuous listening loop (used in navigation mode) ───────────────────────

def listen_for_interrupt(callback) -> None:
    """
    Background listener for interrupt commands during navigation.
    Calls callback(command: str) when user says "stop", "repeat", "help", or "where am i".
    Run this in a daemon thread.

    Args:
        callback: function(command: str) -> None called on recognized keyword
    """
    if _mic is None:
        logger.error("[STT] Interrupt listener aborted (no mic).")
        return

    KEYWORDS = ["stop", "repeat", "help", "where am i"]

    logger.info("[STT] Interrupt listener active.")
    while True:
        with _mic_lock:
            with _mic as source:
                try:
                    _recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = _recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = _recognizer.recognize_google(audio).lower()
                    for kw in KEYWORDS:
                        if kw in text:
                            logger.info(f"[STT] Interrupt command: '{kw}'")
                            callback(kw)
                            break
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass
                except sr.RequestError:
                    pass
        # Give main thread a chance to grab the mic
        time.sleep(0.5)
