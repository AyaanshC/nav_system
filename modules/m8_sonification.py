"""
modules/m8_sonification.py

Instant Audio Sonification Engine.
Plays non-blocking audio beeps proportional to the danger level (depth).
Bypasses slow TTS generation for absolute physical safety.
"""

import time
import threading
import winsound
from loguru import logger

class SonificationEngine:
    def __init__(self):
        self.danger_level = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._thread.start()
        logger.info("[Sonification] Engine started (Background Thread).")

    def set_danger_level(self, level: float):
        """
        Updates the current danger level.
        Args:
            level: Float between 0.0 (safe) and 1.0 (imminent collision)
        """
        self.danger_level = max(0.0, min(1.0, level))

    def stop(self):
        self._running = False

    def _beep_loop(self):
        """Background thread loop that plays beeps."""
        while self._running:
            dl = self.danger_level
            
            if dl <= 0.1:
                # Safe: sleep to prevent CPU spin
                time.sleep(0.1)
                continue
                
            # Danger level 0.1 to 1.0 mapping to beep frequency and delay
            # Pitch: 1000Hz to 2500Hz
            # Delay: 0.8s to 0.1s
            freq = int(1000 + (dl * 1500))
            duration = 100 # ms
            
            # Inverse relationship: higher danger = smaller delay
            delay = 0.9 - (dl * 0.8) 
            delay = max(0.05, delay)
            
            winsound.Beep(freq, duration)
            time.sleep(delay)
