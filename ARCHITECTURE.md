# Architectural Decision Memo

## 1. Discovered Entry Points and Existing Interfaces
- **Repository Topology**: The system is a standalone Python application, not a ROS package. It is heavily modularized within the `modules/` directory.
- **Camera Stream**: Module 1 captures monocular RGB frames via an IP webcam MJPEG stream (`modules/m2_phone_stream.py`), using a threaded background loop to push BGR numpy frames to a queue. There is no physical RGB-D camera available.
- **Localization/SLAM**: Localization (`modules/m2_slam.py`) primarily uses DINOv2 + FAISS image retrieval as a robust fallback (instead of full ORB-SLAM3). It expects monocular BGR frames and returns a topological node ID (a string).
- **Graph/Map Representation**: The topological map (`osmag.json`) stores semantic labels, node IDs, and edge definitions.

## 2. Interface Contract
- **Camera Data**: `modules/m2_phone_stream.py` provides a `.get_frame()` method returning `(frame_id, numpy BGR frame)`. We will inject a mock camera when the `--mock-camera` flag is used.
- **Pose Updates**: `modules/m2_slam.py` `.process_frame()` returns the current `node_id`. A thread-safe state variable will track this string identifier. When `--mock-pose` is used, we can simulate node transitions over time.
- **System Communication**: Without ROS, communication will rely on Python's `threading` and thread-safe data structures like `queue.Queue` or shared dictionaries protected by `threading.Lock`.

## 3. Depth Anything V2 Placement
- **Location**: Depth Anything V2 will be implemented in `src/perception/depth_estimator.py`.
- **Pipeline Integration**: The camera stream loop will pass the latest frame to `depth_estimator.py` to produce a depth map. The bounding box depth calculation `get_bounding_box_distance(bbox, depth_map)` will be a utility function used alongside YOLO to estimate object distances and trigger immediate hazards.

## 4. Concurrency Strategy
- **Perception Loop**: Runs in its own thread to ensure inference (Depth Anything V2 + YOLO) runs as fast as possible (~5-10 Hz) without blocking the rest of the application.
- **Reasoning/VLM Loop**: VLM API calls (which can take >1s) will run asynchronously in a separate thread.
- **Audio Output Loop**: An asynchronous or dedicated thread queue for TTS. Preemptive emergency messages (from the perception thread identifying immediate hazards) will clear this queue and play immediately.
- **Main/Orchestrator**: The master orchestrator (`src/navigation_manager_node.py`) coordinates state sharing, manages thread lifecycles, handles voice inputs, path planning logic when goals change, and monitors the overall system health.
