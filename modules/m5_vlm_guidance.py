"""
modules/m5_vlm_guidance.py

Real-time voice guidance engine with state-of-the-art improvements:

  1. Narrative Memory: maintains a rolling natural-language summary of
     the user's recent journey (last 5 node transitions). Injected into
     every VLM prompt so Moondream understands the trajectory context.

  2. 2-Frame Sliding Window: passes the previous frame alongside the
     current one, giving the VLM a sense of motion direction.

  3. Edge-aware instructions: incorporates turn-by-turn descriptions
     stored on graph edges during scan walk.

  4. Smart caching: only re-queries when node changed, detections changed,
     or cache is stale — prevents redundant API calls.
"""

import os
import base64
import time
import numpy as np
import cv2
from collections import deque
from openai import OpenAI
from loguru import logger


client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

SYSTEM_PROMPT = """You are a real-time navigation assistant for a visually impaired person.

You receive:
  - The user's recent journey (where they've been)
  - Their current location in the building
  - The next navigation step
  - Detected nearby objects and distances
  - Depth sensor readings (left/center/right)
  - Two camera images: previous moment and right now

Your job: Generate ONE short spoken instruction (max 20 words) for RIGHT NOW.
Rules:
  - Safety first: warn about CLOSE obstacles before giving direction
  - Be specific about direction: left, right, straight, stop, slow down
  - Reference visible landmarks when helpful ("past the elevator", "through the glass door")
  - Do NOT say "I see" or "the camera shows" — just give the instruction
  - Sound calm, natural, and confident"""


# ── Narrative Memory ───────────────────────────────────────────────────────────

class NarrativeMemory:
    """
    Maintains a rolling natural-language summary of the user's journey.
    Updated each time the user transitions to a new node.

    Example output:
    "Started at main entrance. Walked into wide corridor. Passed elevator area.
     Turned into narrower hallway. Now at seating area near windows."
    """

    MAX_ENTRIES = 6  # Keep last 6 transitions in memory

    def __init__(self):
        self._entries: deque[str] = deque(maxlen=self.MAX_ENTRIES)
        self._last_node: str | None = None

    def update(self, node_id: str, node_name: str) -> None:
        """Record a node transition into the narrative."""
        if node_id == self._last_node:
            return
        if not self._entries:
            self._entries.append(f"Started at: {node_name}")
        else:
            self._entries.append(f"→ {node_name}")
        self._last_node = node_id

    def get_text(self) -> str:
        """Return the narrative as a single readable string."""
        if not self._entries:
            return "Journey just started."
        return " ".join(self._entries)

    def reset(self):
        """Clear memory (call when a new navigation goal is set)."""
        self._entries.clear()
        self._last_node = None


# ── Guidance Engine ────────────────────────────────────────────────────────────

class GuidanceEngine:
    """
    Generates voice guidance instructions by querying Moondream with
    multi-modal context (2 frames + text narrative). Includes smart
    caching to avoid redundant API calls.
    """

    def __init__(self, max_cache_age_sec: float = 3.0):
        """
        Args:
            max_cache_age_sec: Max seconds to reuse a cached instruction
                               before forcing a new VLM query.
        """
        self._last_instruction = ""
        self._last_query_time = 0.0
        self._max_cache_age = max_cache_age_sec
        self._last_node: str | None = None
        self._last_detection_hash = ""

        # Previous frame buffer for 2-frame sliding window
        self._prev_frame: np.ndarray | None = None

        # Narrative memory — tracks the user's journey
        self.narrative = NarrativeMemory()

    def should_query(
        self,
        current_node: str,
        detection_hash: str,
        node_just_changed: bool = False,
        force: bool = False
    ) -> bool:
        """
        Determine if a new VLM query is needed.

        Triggers re-query if:
          - Node just changed (most important — transition moment)
          - Detections changed (new obstacle appeared)
          - Cache is older than max_cache_age
          - force=True
        """
        now = time.time()
        detection_changed = detection_hash != self._last_detection_hash
        cache_stale = (now - self._last_query_time) > self._max_cache_age

        return (force or node_just_changed or detection_changed or cache_stale)

    def generate_instruction(
        self,
        frame_bgr: np.ndarray,
        current_node: str,
        current_node_name: str,
        next_step_instruction: str,
        path_nl_description: str,
        yolo_text: str,
        depth_text: str,
        detection_hash: str,
        node_just_changed: bool = False,
        edge_description: str = "",
    ) -> str:
        """
        Core method. Assembles prompt with narrative memory + 2-frame window,
        calls Moondream, returns instruction string.
        Returns cached result if no re-query is needed.

        Args:
            frame_bgr:            Current camera frame (BGR)
            current_node:         OSMAG node ID
            current_node_name:    Human-readable name of current node
            next_step_instruction: From PathPlanner
            path_nl_description:  Full path description (for context)
            yolo_text:            From YOLODetector.format_for_prompt()
            depth_text:           From DepthEstimator result["prompt_text"]
            detection_hash:       For caching comparison
            node_just_changed:    True if user just entered this node (triggers immediate re-query)
            edge_description:     Turn-by-turn instruction from osmag.json edge

        Returns:
            Spoken instruction string (≤20 words)
        """
        # Update narrative memory whenever we're called with a node
        self.narrative.update(current_node, current_node_name)

        if not self.should_query(current_node, detection_hash, node_just_changed):
            return self._last_instruction

        # ── Encode current frame at 320px (optimal for Moondream) ─────────────
        h, w = frame_bgr.shape[:2]
        new_w, new_h = 320, int(h * 320 / w)
        small_curr = cv2.resize(frame_bgr, (new_w, new_h))
        _, buf_curr = cv2.imencode(".jpg", small_curr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        b64_curr = base64.b64encode(buf_curr.tobytes()).decode("utf-8")

        # ── Build content list (2-frame window if previous frame available) ───
        # Use the most informative next-step directive (edge description beats generic)
        direction = edge_description if edge_description else next_step_instruction

        user_text = (
            f"Your recent journey: {self.narrative.get_text()}\n\n"
            f"Current location: {current_node_name}\n"
            f"Next step: {direction}\n"
            f"{yolo_text}\n"
            f"{depth_text}\n\n"
            f"Generate ONE spoken navigation instruction for right now:"
        )

        content: list[dict] = [{"type": "text", "text": user_text}]

        # Add previous frame first (gives sense of direction/motion)
        if self._prev_frame is not None:
            prev_small = cv2.resize(self._prev_frame, (new_w, new_h))
            _, buf_prev = cv2.imencode(".jpg", prev_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_prev = base64.b64encode(buf_prev.tobytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_prev}", "detail": "low"}
            })

        # Add current frame
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_curr}", "detail": "low"}
        })

        try:
            response = client.chat.completions.create(
                model="qwen2.5vl:3b",
                max_tokens=50,
                temperature=0.15,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": content}
                ]
            )
            instruction = response.choices[0].message.content.strip()
            instruction = instruction.strip('"').strip("'")

            # Update cache state
            self._last_instruction = instruction
            self._last_query_time = time.time()
            self._last_node = current_node
            self._last_detection_hash = detection_hash

            logger.info(f"[VLM] Instruction: {instruction}")

        except Exception as e:
            logger.error(f"[VLM] Query failed: {e}")
            # Graceful degradation: fall back to edge description → BFS step
            instruction = (
                edge_description or next_step_instruction or self._last_instruction
            )

        # Store current frame as previous for next call
        self._prev_frame = frame_bgr.copy()

        return instruction
