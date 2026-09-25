import json
import yaml
from typing import Dict, List, Optional
from loguru import logger
import os

class SemanticGraph:
    """
    Topological Graph Representation (OSMAG-compliant).
    """
    def __init__(self, map_path: str):
        self.map_path = map_path
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []

        self._load_graph()

    def _load_graph(self):
        """Loads JSON or YAML indoor maps."""
        if not os.path.exists(self.map_path):
            logger.warning(f"[Graph] Map file {self.map_path} not found. Starting with empty graph.")
            return

        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                if self.map_path.endswith('.json'):
                    data = json.load(f)
                elif self.map_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    logger.error(f"[Graph] Unsupported file format: {self.map_path}")
                    return

            if "nodes" in data:
                if isinstance(data["nodes"], list):
                    # Convert list to dict indexed by id
                    self.nodes = {str(n["id"]): n for n in data["nodes"]}
                elif isinstance(data["nodes"], dict):
                    self.nodes = data["nodes"]

            self.edges = data.get("edges", [])

            logger.info(f"[Graph] Loaded {len(self.nodes)} nodes and {len(self.edges)} edges from {self.map_path}")
        except Exception as e:
            logger.error(f"[Graph] Failed to load map: {e}")

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(str(node_id))

    def mark_node_out_of_service(self, node_id: str):
        """Dynamically marks a node as out of service."""
        node = self.get_node(node_id)
        if node:
            if "status" not in node:
                node["status"] = {}
            node["status"]["in_service"] = False
            logger.info(f"[Graph] Node {node_id} marked as out of service.")

    def get_neighbors(self, node_id: str) -> List[dict]:
        """Returns list of neighboring edges for a given node."""
        node_id_str = str(node_id)
        neighbors = []
        for edge in self.edges:
            if str(edge.get("from")) == node_id_str:
                neighbors.append(edge)
            # Assuming bidirectional graph by default if 'to' matches
            elif str(edge.get("to")) == node_id_str:
                # Reverse edge for neighbor perspective
                rev_edge = edge.copy()
                rev_edge["from"] = edge["to"]
                rev_edge["to"] = edge["from"]
                neighbors.append(rev_edge)

        # Also check internal node 'neighbors' field for custom directional attributes
        node = self.get_node(node_id_str)
        if node and "neighbors" in node:
            for neighbor_id, direction in node["neighbors"].items():
                neighbors.append({
                    "from": node_id_str,
                    "to": str(neighbor_id),
                    "direction": direction
                })
        return neighbors

def create_seed_campus_graph(output_path: str = "campus_graph.json"):
    """
    Creates a seed campus graph file modeling multi-floor transitions.
    """
    graph_data = {
        "nodes": [
            {
                "id": "1421",
                "name": "Room 1421",
                "level": 14,
                "areaType": "room",
                "status": {"in_service": True}
            },
            {
                "id": "14_elevator_1",
                "name": "14 elevator 1",
                "level": 14,
                "areaType": "elevator",
                "status": {"in_service": False}  # Out of service!
            },
            {
                "id": "14_stairs_1",
                "name": "14 stairs 1",
                "level": 14,
                "areaType": "stairs",
                "status": {"in_service": True}
            },
            {
                "id": "15_stairs_1",
                "name": "15 stairs 1",
                "level": 15,
                "areaType": "stairs",
                "status": {"in_service": True}
            },
            {
                "id": "15_elevator_1",
                "name": "15 elevator 1",
                "level": 15,
                "areaType": "elevator",
                "status": {"in_service": True}
            },
            {
                "id": "1521",
                "name": "Room 1521",
                "level": 15,
                "areaType": "room",
                "status": {"in_service": True}
            }
        ],
        "edges": [
            {"from": "1421", "to": "14_elevator_1", "description": "straight"},
            {"from": "1421", "to": "14_stairs_1", "description": "left"},
            {"from": "14_stairs_1", "to": "15_stairs_1", "description": "up"},
            {"from": "14_elevator_1", "to": "15_elevator_1", "description": "up"},
            {"from": "15_stairs_1", "to": "1521", "description": "right"},
            {"from": "15_elevator_1", "to": "1521", "description": "straight"}
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)
    logger.info(f"[Graph] Created seed campus graph at {output_path}")

if __name__ == "__main__":
    create_seed_campus_graph()
