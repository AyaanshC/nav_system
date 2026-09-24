import argparse
import time
import threading
import queue
import numpy as np
from loguru import logger
import sys

# Import new modules
from src.perception.depth_estimator import DepthEstimator
from src.perception.yolo_detector import YOLOWorldDetector
from src.navigation.semantic_graph import SemanticGraph
from src.navigation.path_planner import SemanticConstrainedPlanner
from src.reasoning.llm_route_explainer import LLMRouteExplainer
from src.reasoning.vlm_guidance_node import VLMGuidanceNode
from src.interaction.voice_input import VoiceInput
from src.interaction.tts_output import TTSOutput

# Import required Module 1 components where possible
try:
    from modules.m2_phone_stream import PhoneStream
    from modules.m2_slam import SLAMLocalizer
except ImportError:
    PhoneStream = None
    SLAMLocalizer = None


class NavigationManager:
    """
    Master Orchestrator.
    Links all modules together: Voice Input -> BFS Planner -> Route Summary ->
    Perception & Grounded Guidance Loop -> Audio Feedback.
    """
    def __init__(self, mock_camera: bool = False, mock_pose: bool = False):
        self.mock_camera = mock_camera
        self.mock_pose = mock_pose

        # State variables
        self.running = False
        self.current_node = "1421" if mock_pose else None
        self.goal_node = None
        self.path_nodes = []
        self.latest_frame = None

        # Initialize modules
        logger.info("Initializing Graph and Planner...")
        self.graph = SemanticGraph("campus_graph.json")
        self.planner = SemanticConstrainedPlanner(self.graph)

        logger.info("Initializing Reasoning Modules...")
        self.explainer = LLMRouteExplainer(self.graph)
        self.vlm_guidance = VLMGuidanceNode()

        logger.info("Initializing Perception Modules...")
        self.depth_estimator = DepthEstimator(device="cpu" if mock_camera else None)
        self.yolo = YOLOWorldDetector(device="cpu" if mock_camera else None)

        logger.info("Initializing Interaction Modules...")
        self.stt = VoiceInput()
        self.tts = TTSOutput()

        # Module 1 specific init
        if not self.mock_camera and PhoneStream:
            # Requires config/settings.yaml phone_stream_url usually
            self.stream = PhoneStream("0")
        else:
            self.stream = None

        if not self.mock_pose and SLAMLocalizer:
            self.slam = SLAMLocalizer("maps/default")
        else:
            self.slam = None

    def _mock_camera_loop(self):
        """Simulates camera frames."""
        while self.running:
            self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)  # 10Hz

    def _mock_pose_loop(self):
        """Simulates user walking along the path."""
        while self.running:
            time.sleep(5.0)
            if self.path_nodes and self.current_node in self.path_nodes:
                idx = self.path_nodes.index(self.current_node)
                if idx + 1 < len(self.path_nodes):
                    self.current_node = self.path_nodes[idx + 1]
                    logger.info(f"[Mock SLAM] Moved to node: {self.current_node}")

    def _perception_loop(self):
        """Runs Depth + YOLO at ~10 Hz and handles VLM logic asynchronously."""
        while self.running:
            start_time = time.time()

            frame = None
            if self.mock_camera:
                frame = self.latest_frame
            elif self.stream:
                _, frame = self.stream.get_frame(timeout=1.0)

            if frame is None:
                time.sleep(0.1)
                continue

            # Perception
            depth_map = self.depth_estimator.estimate(frame)
            detections, changed, immediate_hazard = self.yolo.detect(frame, depth_map, self.depth_estimator)

            # Preemptive Safety Alert
            if immediate_hazard:
                # Get the hazard name
                hazards = [d for d in detections if d["direction"] == "Center" and d["distance"] < 1.5]
                if hazards:
                    closest = min(hazards, key=lambda x: x["distance"])
                    self.tts.speak_alert(f"Caution: {closest['class']} {closest['distance']:.1f} meters directly ahead.")
                    # Prevent VLM from overriding this immediately by sleeping briefly
                    time.sleep(1.0)

            # Guidance update via VLM
            if self.path_nodes and self.current_node:
                idx = self.path_nodes.index(self.current_node)
                next_node = self.path_nodes[idx + 1] if idx + 1 < len(self.path_nodes) else self.current_node

                # Use string names for guidance
                curr_name = self.graph.get_node(self.current_node).get("name", self.current_node) if self.graph.get_node(self.current_node) else self.current_node
                next_name = self.graph.get_node(next_node).get("name", next_node) if self.graph.get_node(next_node) else next_node

                # Calculate detection hash (simplified)
                det_hash = str(len(detections)) + ("_".join([d['class'] for d in detections]))

                if self.vlm_guidance.should_reprompt(self.current_node, next_node, det_hash):
                    # In a real app this would be in another async thread so it doesn't block the perception loop
                    # For simplicity, we call it directly here.
                    guidance = self.vlm_guidance.generate_guidance(frame, curr_name, next_name, detections, det_hash)
                    if guidance:
                        self.tts.speak(guidance)

            # Ensure ~10Hz loop
            elapsed = time.time() - start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    def start(self):
        self.running = True
        logger.info("Starting Navigation System...")

        # Start hardware streams
        if not self.mock_camera and self.stream:
            self.stream.start()

        # Start threads
        if self.mock_camera:
            threading.Thread(target=self._mock_camera_loop, daemon=True).start()
        if self.mock_pose:
            threading.Thread(target=self._mock_pose_loop, daemon=True).start()

        threading.Thread(target=self._perception_loop, daemon=True).start()

        # Main UI/Interaction Loop
        try:
            self.tts.speak("System ready. Tell me your destination.")
            time.sleep(3) # Give TTS time to speak

            while self.running:
                # 1. Listen for goal
                spoken = self.stt.listen_for_destination()
                if not spoken:
                    time.sleep(1)
                    continue

                # 2. Simple goal resolution
                logger.info(f"Resolving goal for: {spoken}")
                # Mock resolution: if they say 1521, set it
                target_id = None
                for n_id, node in self.graph.nodes.items():
                    if n_id in spoken or node.get("name", "").lower() in spoken.lower():
                        target_id = n_id
                        break

                if not target_id:
                    self.tts.speak("Destination not recognized. Please try again.")
                    continue

                self.goal_node = target_id

                # 3. Path Planning
                if self.current_node:
                    path_dict = self.planner.find_path(self.current_node, self.goal_node)
                    self.path_nodes = path_dict.get("path", [])

                    if not self.path_nodes:
                        self.tts.speak(f"Cannot find a route to that destination.")
                        continue

                    # 4. Summarize Route
                    summary = self.explainer.summarize_route(self.path_nodes)
                    logger.info(f"Route Summary: {summary}")
                    self.tts.speak(summary)
                else:
                    self.tts.speak("Current location unknown. Cannot plan route.")

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.stop()

    def stop(self):
        self.running = False
        if not self.mock_camera and self.stream:
            self.stream.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-camera", action="store_true", help="Run without physical camera")
    parser.add_argument("--mock-pose", action="store_true", help="Run without RTAB-Map SLAM")
    args = parser.parse_args()

    manager = NavigationManager(mock_camera=args.mock_camera, mock_pose=args.mock_pose)
    manager.start()
