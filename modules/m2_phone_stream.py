"""
modules/m2_phone_stream.py

Connects to the phone's IP Webcam MJPEG stream.
Puts frames into a thread-safe queue for downstream consumers.

Phone setup (Android IP Webcam app):
  - Start server on phone
  - Set resolution to 640x480 (lower = faster WiFi transfer)
  - Note the IP:port shown on screen (e.g. 192.168.1.5:8080)
  - URL format: http://192.168.1.5:8080/video
"""

import cv2
import threading
import queue
import time
from loguru import logger


class PhoneStream:
    """
    Manages connection to IP Webcam MJPEG stream.
    Provides get_frame() which returns the latest BGR frame.
    Automatically reconnects on dropout.
    """

    def __init__(self, url: str, max_queue_size: int = 3):
        """
        Args:
            url: IP Webcam stream URL e.g. 'http://192.168.1.5:8080/video'
            max_queue_size: discard old frames if queue fills (always keep freshest)
        """
        self.url = url
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._start_time = 0.0

    def start(self):
        """Start background capture thread."""
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Stream] Started capture from {self.url}")

    def stop(self):
        """Stop capture thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("[Stream] Stopped.")

    def get_frame(self, timeout: float = 2.0):
        """
        Returns (frame_id, numpy BGR frame) or (None, None) on timeout.
        frame_id increments with every captured frame — use for N-th frame logic.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def _capture_loop(self):
        # Convert stream URL to integer if it's just a number (e.g., "0" or "1" for Iriun Webcam)
        source = int(self.url) if str(self.url).isdigit() else self.url

        while self._running:
            if isinstance(source, int):
                # Force DirectShow backend for local/virtual cameras on Windows to avoid MSMF errors
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)
                
            if not cap.isOpened():
                logger.error(f"[Stream] Cannot open {self.url}. Retrying in 3s...")
                time.sleep(3)
                continue

            logger.info("[Stream] Connected to phone camera.")
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("[Stream] Frame read failed. Reconnecting...")
                    break

                self._frame_count += 1

                # Drop oldest frame if queue full (always keep freshest)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

                self._queue.put((self._frame_count, frame))

            cap.release()
            if self._running:
                logger.info("[Stream] Reconnecting in 2s...")
                time.sleep(2)

    @property
    def fps(self) -> float:
        """Approximate average FPS since start."""
        elapsed = time.time() - self._start_time
        return self._frame_count / max(1.0, elapsed)

    @property
    def frame_count(self) -> int:
        return self._frame_count
