import time
from typing import List, Dict, Tuple
from loguru import logger
import base64
import cv2
import numpy as np

class VLMGuidanceNode:
    """
    Real-Time Perception-Grounded VLM Guidance.
    Formulates grounded prompts matching paper Section 3.2.3.
    """
    def __init__(self, rate_limit_sec: float = 3.0, vlm_client=None, model: str = "qwen2.5-vl:3b"):
        self.rate_limit_sec = rate_limit_sec
        self.last_prompt_time = 0.0
        self.last_node = None
        self.last_target = None
        self.last_detection_hash = ""

        # If client not provided, it will use local Ollama
        import openai
        if vlm_client:
            self.vlm_client = vlm_client
        else:
            logger.info("[VLM] Initializing local Ollama client.")
            self.vlm_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model

    def _encode_image(self, frame_bgr: np.ndarray) -> str:
        """Encodes frame to base64 for API transmission."""
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def should_reprompt(self, curr_node: str, next_node: str, detection_hash: str) -> bool:
        """
        Rate limiting: Re-prompt only when approaching a new node, changing orientation,
        or when new obstacles block the path.
        """
        now = time.time()

        if now - self.last_prompt_time < self.rate_limit_sec:
            return False

        if curr_node != self.last_node:
            return True

        if next_node != self.last_target:
            return True

        if detection_hash != self.last_detection_hash:
            return True

        # Reprompt after long timeout even if nothing changed
        if now - self.last_prompt_time > 10.0:
            return True

        return False

    def generate_guidance(self,
                          frame_bgr: np.ndarray,
                          curr_node_name: str,
                          next_node_name: str,
                          detections: List[Dict],
                          detection_hash: str) -> str:
        """
        Generates 1-sentence concise, safe directional cue.
        """
        # Format detections
        if not detections:
            detected_str = "None"
        else:
            det_parts = [f"{d['class']} ({d['distance']:.1f}m, {d['direction']})" for d in detections]
            detected_str = ", ".join(det_parts)

        prompt = f"""Current location: {curr_node_name}. Target waypoint: {next_node_name}.
Detected objects: {detected_str}.
Task: Produce a 1-sentence concise, safe directional cue for a visually impaired user."""

        now = time.time()
        self.last_prompt_time = now
        self.last_node = curr_node_name
        self.last_target = next_node_name
        self.last_detection_hash = detection_hash

        logger.debug(f"[VLM] Grounded Prompt:\n{prompt}")

        if self.vlm_client:
            try:
                base64_image = self._encode_image(frame_bgr)
                # Note: this is a generic OpenAI format, for local ollama, might need specific adjustment
                # but following typical vision API format.
                response = self.vlm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"[VLM] Call failed: {e}")
                return self._deterministic_fallback(curr_node_name, next_node_name, detections)
        else:
            # Deterministic fallback derived directly from YOLO + Depth
            return self._deterministic_fallback(curr_node_name, next_node_name, detections)

    def _deterministic_fallback(self, curr: str, nxt: str, detections: List[Dict]) -> str:
        """Clean degradation: template-based safety cues."""
        if not detections:
            return f"Path clear. Proceed towards {nxt}."

        # Find closest center obstacle
        center_hazards = [d for d in detections if d["direction"] == "Center"]
        if center_hazards:
            closest = min(center_hazards, key=lambda x: x["distance"])
            return f"Caution: {closest['class']} directly ahead at {closest['distance']:.1f} meters."

        # General guidance
        return f"Proceed towards {nxt}, adjusting for obstacles nearby."
