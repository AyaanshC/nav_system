import heapq
from typing import List, Dict, Tuple
from loguru import logger
from src.navigation.semantic_graph import SemanticGraph

class SemanticConstrainedPlanner:
    """
    Semantic Constrained BFS/Dijkstra Planner.
    Finds the shortest node sequence while respecting semantic constraints (e.g., out-of-service elevators).
    """
    def __init__(self, graph: SemanticGraph):
        self.graph = graph

    def _get_node_cost(self, node_id: str) -> float:
        """Assigns safety/semantic weight based on areaType and status."""
        node = self.graph.get_node(node_id)
        if not node:
            return 1000.0  # High cost for unknown nodes

        # Check if out of service
        status = node.get("status", {})
        if not status.get("in_service", True):
            return 1000.0  # Extremely high cost for out of service areas

        area_type = node.get("areaType", "corridor")
        cost_map = {
            "corridor": 1.0,
            "room": 1.0,
            "open_area": 1.0,
            "entrance": 2.0,
            "stairs": 1.5,     # Slightly higher cost than corridor but usable
            "elevator": 1.2    # Prefer elevators if in service
        }

        return cost_map.get(area_type, 1.0)

    def find_path(self, start_node: str, goal_node: str) -> Dict:
        """
        Dijkstra search from start_node to goal_node respecting semantic constraints.

        Args:
            start_node: OSMAG node ID of current location
            goal_node: OSMAG node ID of destination

        Returns:
            Dictionary with structured path: {"path": ["node1", "node2", ...]}
        """
        start_node = str(start_node)
        goal_node = str(goal_node)

        if not self.graph.get_node(start_node) or not self.graph.get_node(goal_node):
            logger.error(f"[Planner] Start or Goal node not in graph: {start_node} -> {goal_node}")
            return {"path": []}

        if start_node == goal_node:
            return {"path": [start_node]}

        # Priority queue: (accumulated_cost, current_node, path_so_far)
        queue = [(0.0, start_node, [start_node])]
        costs = {start_node: 0.0}

        while queue:
            current_cost, current_node, path = heapq.heappop(queue)

            if current_node == goal_node:
                return {"path": path}

            for edge in self.graph.get_neighbors(current_node):
                neighbor = str(edge.get("to"))
                if not neighbor:
                    continue

                neighbor_cost = self._get_node_cost(neighbor)

                # If neighbor is practically impassable, skip
                if neighbor_cost >= 1000.0:
                    continue

                new_cost = current_cost + neighbor_cost

                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

        logger.warning(f"[Planner] No valid path found from {start_node} to {goal_node}")
        return {"path": []}
