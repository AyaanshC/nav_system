import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import hashlib
from loguru import logger

# Import modules
from src.perception.depth_estimator import DepthEstimator
from src.perception.yolo_detector import YOLOWorldDetector
from src.navigation.semantic_graph import SemanticGraph
from src.navigation.path_planner import SemanticConstrainedPlanner
from src.reasoning.llm_route_explainer import LLMRouteExplainer
from src.reasoning.vlm_guidance_node import VLMGuidanceNode
from src.interaction.tts_output import TTSOutput

class NavigationROSNode(Node):
    """
    ROS 2 Orchestrator Node for Multimodal Navigation System.
    """
    def __init__(self):
        super().__init__('multimodal_navigation_node')

        self.bridge = CvBridge()

        # ROS 2 Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            String,
            '/rtabmap/localization_pose',
            self.pose_callback,
            10
        )

        self.voice_goal_sub = self.create_subscription(
            String,
            '/interaction/voice_goal',
            self.voice_goal_callback,
            10
        )

        # ROS 2 Publishers
        self.guidance_pub = self.create_publisher(String, '/interaction/guidance', 10)
        self.alert_pub = self.create_publisher(String, '/interaction/alert', 10)

        # State
        self.current_node = None
        self.goal_node = None
        self.path_nodes = []
        self._obstacle_history = {}

        # Initialize Core Systems
        self.get_logger().info("Initializing Navigation Subsystems...")
        try:
            self.graph = SemanticGraph("campus_graph.json")
            self.planner = SemanticConstrainedPlanner(self.graph)
            self.explainer = LLMRouteExplainer(self.graph)
            self.vlm_guidance = VLMGuidanceNode()
            self.depth_estimator = DepthEstimator()

            # Use ONNX if available for optimization, otherwise fallback to PT
            self.yolo = YOLOWorldDetector(model_path="yolov8s-world.onnx")

            # Local audio fallback for ease of use if not subscribed
            self.tts = TTSOutput()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize subsystems: {e}")

    def pose_callback(self, msg: String):
        """Updates current topological semantic node ID based on SLAM localization."""
        new_node = msg.data
        if self.current_node != new_node:
            self.current_node = new_node
            self.get_logger().info(f"Localized at node: {self.current_node}")

    def voice_goal_callback(self, msg: String):
        """Handles new voice-commanded destination."""
        spoken = msg.data
        self.get_logger().info(f"Received goal request: {spoken}")

        target_id = None
        for n_id, node in self.graph.nodes.items():
            if n_id in spoken or node.get("name", "").lower() in spoken.lower():
                target_id = n_id
                break

        if not target_id:
            self.tts.speak("Destination not recognized.")
            return

        self.goal_node = target_id

        if self.current_node:
            self._replan()
        else:
            self.tts.speak("Current location unknown. Cannot plan route.")

    def _replan(self):
        """Recalculates route and explains it."""
        path_dict = self.planner.find_path(self.current_node, self.goal_node)
        self.path_nodes = path_dict.get("path", [])

        if not self.path_nodes:
            self.tts.speak("Cannot find a valid alternate route. Destination unreachable.")
        else:
            summary = self.explainer.summarize_route(self.path_nodes)
            self.get_logger().info(f"New Route Summary: {summary}")
            self.tts.speak(f"New route calculated. {summary}")

    def image_callback(self, msg: Image):
        """Main perception pipeline triggered by camera frames."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 1. Perception
        depth_map = self.depth_estimator.estimate(frame)
        detections, changed, immediate_hazard = self.yolo.detect(frame, depth_map, self.depth_estimator)

        # 2. Dynamic Graph & Safety Alerts
        current_detected_classes = set()
        if immediate_hazard:
            hazards = [d for d in detections if d["direction"] == "Center" and d["distance"] < 1.5]
            if hazards:
                closest = min(hazards, key=lambda x: x["distance"])
                alert_text = f"Caution: {closest['class']} {closest['distance']:.1f} meters directly ahead."

                # Publish ROS alert and speak locally
                self.alert_pub.publish(String(data=alert_text))
                self.tts.speak_alert(alert_text)

                # Dynamic replanning tracker
                hazard_class = closest['class']
                current_detected_classes.add(hazard_class)
                self._obstacle_history[hazard_class] = self._obstacle_history.get(hazard_class, 0) + 1

                # 20 consecutive frames -> permanent block
                if self._obstacle_history[hazard_class] >= 20 and self.path_nodes and self.current_node:
                    idx = self.path_nodes.index(self.current_node)
                    if idx + 1 < len(self.path_nodes):
                        next_node = self.path_nodes[idx + 1]
                        self.get_logger().warn(f"Persistent obstacle '{hazard_class}' detected. Marking {next_node} out of service.")
                        self.graph.mark_node_out_of_service(next_node)
                        self._replan()
                        self._obstacle_history[hazard_class] = 0

        # Decay unseen obstacles
        for cls in list(self._obstacle_history.keys()):
            if cls not in current_detected_classes:
                self._obstacle_history[cls] = 0

        # 3. Grounded VLM Guidance
        if self.path_nodes and self.current_node:
            idx = self.path_nodes.index(self.current_node)
            next_node = self.path_nodes[idx + 1] if idx + 1 < len(self.path_nodes) else self.current_node

            curr_name = self.graph.get_node(self.current_node).get("name", self.current_node) if self.graph.get_node(self.current_node) else self.current_node
            next_name = self.graph.get_node(next_node).get("name", next_node) if self.graph.get_node(next_node) else next_node

            det_hash = str(len(detections)) + ("_".join([d['class'] for d in detections]))

            if self.vlm_guidance.should_reprompt(self.current_node, next_node, det_hash):
                guidance = self.vlm_guidance.generate_guidance(frame, curr_name, next_name, detections, det_hash)
                if guidance:
                    self.guidance_pub.publish(String(data=guidance))
                    self.tts.speak(guidance)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
