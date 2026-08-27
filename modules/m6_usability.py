"""
modules/m6_usability.py

Flask web server providing:
  - Caregiver dashboard (monitor guidance, location, obstacles, adjust TTS rate)
  - Session logger (writes all guidance events to JSON incrementally)
  - REST API consumed by main.py for settings sync

Runs in a background daemon thread — does not block the main navigation loop.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from threading import Thread
from loguru import logger


app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "web_ui" / "templates"),
    static_folder=str(Path(__file__).parent.parent / "web_ui" / "static")
)
CORS(app)


# ── Shared state (written by main.py, read by web UI via REST) ─────────────────

_state: dict = {
    "current_node":      "unknown",
    "current_node_name": "unknown location",
    "goal_node":         None,
    "goal_name":         None,
    "last_instruction":  "",
    "detections":        [],
    "depth":             {},
    "tts_rate":          "+0%",
    "session_start":     datetime.now().isoformat(),
    "instruction_count": 0,
    "is_navigating":     False,
}


_session_log: list[dict] = []
_log_dir = Path(__file__).parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_session_file = _log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# ── State API (called by main.py) ──────────────────────────────────────────────

def update_state(**kwargs) -> None:
    """Called by main.py to push live navigation data into the web UI state."""
    _state.update(kwargs)


def log_event(event_type: str, data: dict) -> None:
    """
    Log a navigation event to the session JSON file.

    Args:
        event_type: One of "instruction", "node_change", "obstacle",
                    "goal_set", "path_computed", "arrived"
        data:       Event-specific fields to log
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type":      event_type,
        **data
    }
    _session_log.append(entry)
    _state["instruction_count"] = len(
        [e for e in _session_log if e["type"] == "instruction"]
    )

    # Write incrementally so log is readable even if process crashes
    try:
        with open(_session_file, "w", encoding="utf-8") as f:
            json.dump(_session_log, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"[WebUI] Could not write session log: {e}")


def get_tts_rate() -> str:
    """Get current TTS rate from shared state."""
    return _state.get("tts_rate", "+0%")


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Live navigation state for caregiver dashboard."""
    return jsonify(_state)


@app.route("/api/set_tts_rate", methods=["POST"])
def set_tts_rate():
    """Caregiver can adjust TTS speech speed from the web UI."""
    rate = request.json.get("rate", "+0%")
    _state["tts_rate"] = rate
    logger.info(f"[WebUI] TTS rate set to {rate}")
    return jsonify({"ok": True, "rate": rate})


@app.route("/api/log")
def get_log():
    """Return last 50 session events."""
    return jsonify(_session_log[-50:])


@app.route("/api/buildings")
def list_buildings():
    """List all buildings that have a valid osmag.json map."""
    maps_dir = Path("maps")
    if not maps_dir.exists():
        return jsonify([])
    buildings = [d.name for d in maps_dir.iterdir() if (d / "osmag.json").exists()]
    return jsonify(buildings)


@app.route("/api/nodes/<building>")
def list_nodes(building: str):
    """List all nodes in a building's OSMAG graph."""
    osmag_path = Path("maps") / building / "osmag.json"
    if not osmag_path.exists():
        return jsonify({"error": "building not found"}), 404
    with open(osmag_path, encoding="utf-8") as f:
        osmag = json.load(f)
    nodes = [{"id": nid, "name": n["name"]} for nid, n in osmag["nodes"].items()]
    return jsonify(nodes)


@app.route("/api/session_file")
def session_file_path():
    """Return current session log file path."""
    return jsonify({"path": str(_session_file)})


# ── Web UI server runner ───────────────────────────────────────────────────────

def start_web_server(host: str = "0.0.0.0", port: int = 5050) -> None:
    """Start Flask in a background daemon thread. Non-blocking."""
    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = Thread(target=_run, daemon=True)
    t.start()
    logger.info(f"[WebUI] Caregiver dashboard at http://localhost:{port}")
