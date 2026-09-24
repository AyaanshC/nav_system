import speech_recognition as sr
from loguru import logger
import time

class VoiceInput:
    """
    Multimodal User Interaction: STT
    Uses SpeechRecognition to capture destination queries with VAD and timeout.
    """
    def __init__(self, timeout: float = 5.0, phrase_time_limit: float = 5.0):
        self.recognizer = sr.Recognizer()
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

        # Adjust for ambient noise
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            logger.warning(f"[STT] Could not adjust for ambient noise: {e}")

    def listen_for_destination(self) -> str:
        """
        Listens for a voice query.
        Returns the recognized text, or empty string on failure/timeout.
        """
        try:
            with sr.Microphone() as source:
                logger.info("[STT] Listening for destination...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit
                )

            text = self.recognizer.recognize_google(audio)
            logger.info(f"[STT] Recognized: '{text}'")
            return text

        except sr.WaitTimeoutError:
            logger.warning("[STT] Timeout: Destination not heard.")
            return ""
        except sr.UnknownValueError:
            logger.warning("[STT] Unknown value: Could not understand audio.")
            return ""
        except sr.RequestError as e:
            logger.error(f"[STT] API error: {e}")
            return ""
        except Exception as e:
            logger.error(f"[STT] Unexpected error: {e}")
            return ""
