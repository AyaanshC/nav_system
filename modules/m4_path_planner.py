"""
modules/m4_path_planner.py

BFS path planner on OSMAG graph.
LLM semantic path description generator.
Caches description — only re-queries when start/goal change.
"""

import json
import os
import re
import heapq
from collections import deque
from pathlib import Path
from openai import OpenAI
from loguru import logger


client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


class PathPlanner:
    """
    Loads an OSMAG JSON graph and provides:
      - BFS path finding between nodes
      - Voice-friendly goal resolution from spoken words
      - GPT-4o-mini natural language path description (cached)
      - Single next-step instructions during navigation
    """

    def __init__(self, osmag_path: str):
        """
        Args:
            osmag_path: Path to maps/<building>/osmag.json
        """
        self.osmag_path = osmag_path
        self.graph: dict[str, list[str]] = {}   # node_id -> list of neighbor node_ids
        self.nodes: dict[str, dict] = {}         # node_id -> full node dict
        self.edges_map: dict[tuple, str] = {}    # (from_id, to_id) -> edge description
        self._load_graph()

        # Caching for NL description
        self._cached_path_nodes: list[str] = []
        self._cached_nl_description: str = ""
        self._cached_for_start: str | None = None
        self._cached_for_goal: str | None = None

    # ── Graph loading ─────────────────────────────────────────────────────────

    def _load_graph(self):
        with open(self.osmag_path, encoding="utf-8") as f:
            osmag = json.load(f)

        self.nodes = osmag["nodes"]

        # Build adjacency list (bidirectional) and edge description map
        for node_id in self.nodes:
            self.graph[node_id] = []

        for edge in osmag.get("edges", []):
            src, dst = edge["from"], edge["to"]
            if src in self.graph and dst not in self.graph[src]:
                self.graph[src].append(dst)
            if dst in self.graph and src not in self.graph[dst]:
                self.graph[dst].append(src)
            # Store edge description for both directions
            desc = edge.get("description", "")
            if desc:
                self.edges_map[(src, dst)] = desc
                self.edges_map[(dst, src)] = desc

        logger.info(
            f"[PathPlanner] Loaded {len(self.nodes)} nodes, "
            f"{len(osmag.get('edges', []))} edges."
        )

    def reload(self):
        """Reload the graph from disk (call if osmag.json was updated)."""
        self.graph.clear()
        self.nodes.clear()
        self._load_graph()
        self._cached_nl_description = ""

    # ── Path finding ──────────────────────────────────────────────────────────

    def _area_cost(self, node_id: str) -> int:
        """Assigns safety weight based on areaType. Higher cost = less safe."""
        area_type = self.nodes.get(node_id, {}).get("areaType", "corridor")
        cost_map = {
            "corridor": 1,
            "room": 1,
            "open_area": 1,
            "entrance": 2,
            "elevator": 2,
            "stairwell": 10  # Penalize heavily to route around stairs if possible
        }
        return cost_map.get(area_type, 1)

    def find_path(self, start_node: str, goal_node: str) -> list[str]:
        """
        Dijkstra search from start_node to goal_node prioritizing safe paths.
        
        Args:
            start_node: OSMAG node ID of current location
            goal_node:  OSMAG node ID of destination

        Returns:
            List of node IDs forming the path, or empty list if unreachable.
        """
        if start_node not in self.graph or goal_node not in self.graph:
            logger.error(
                f"[PathPlanner] Node not in graph: start={start_node}, goal={goal_node}"
            )
            return []

        if start_node == goal_node:
            return [start_node]

        # Priority queue stores tuples of (accumulated_cost, current_node, path_so_far)
        queue = [(0, start_node, [start_node])]
        
        # Track the lowest cost to reach each node to avoid suboptimal revisiting
        costs = {start_node: 0}

        while queue:
            current_cost, current_node, path = heapq.heappop(queue)

            if current_node == goal_node:
                return path

            for neighbor in self.graph.get(current_node, []):
                # Calculate cost to traverse to neighbor
                neighbor_cost = self._area_cost(neighbor)
                new_cost = current_cost + neighbor_cost

                # If we found a cheaper way to reach neighbor (or it's unvisited)
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

        logger.warning(f"[PathPlanner] No path found: {start_node} → {goal_node}")
        return []

    # ── Goal resolution ───────────────────────────────────────────────────────

    def resolve_goal_node(self, spoken_goal: str) -> str | None:
        """
        Match a spoken goal string (e.g. "restroom") to the best node ID.
        Uses substring word matching — scores each node by how many spoken words appear in its name.

        Args:
            spoken_goal: Transcribed user utterance, lowercased

        Returns:
            Best-matching node_id, or None if no match found
        """
        spoken_lower = spoken_goal.lower()
        candidates: list[tuple[int, str]] = []

        for node_id, node in self.nodes.items():
            name_lower = node["name"].lower()
            # Also check area type and landmarks
            landmarks_text = " ".join(node.get("landmarks", [])).lower()
            full_text = f"{name_lower} {node.get('areaType', '')} {landmarks_text}"

            words = spoken_lower.split()
            score = sum(
                1 for w in words
                if len(w) > 2 and re.search(r'\b' + re.escape(w) + r'\b', full_text)
            )
            if score > 0:
                candidates.append((score, node_id))

        if not candidates:
            logger.warning(f"[PathPlanner] No node matched '{spoken_goal}'")
            return None

        candidates.sort(reverse=True)
        best_id = candidates[0][1]
        logger.info(
            f"[PathPlanner] Resolved '{spoken_goal}' → node '{best_id}' "
            f"({self.nodes[best_id]['name']})"
        )
        return best_id

    # ── Natural language description ──────────────────────────────────────────

    def get_nl_description(
        self,
        start_node: str,
        goal_node: str,
        path_nodes: list[str]
    ) -> str:
        """
        Generate natural language path description via GPT-4o-mini.
        CACHED — only calls API if start or goal changed.

        Args:
            start_node:  Starting node ID
            goal_node:   Destination node ID
            path_nodes:  Ordered list of node IDs along the path

        Returns:
            Multi-step spoken directions string
        """
        # Return cached result if start/goal unchanged
        if (start_node == self._cached_for_start
                and goal_node == self._cached_for_goal
                and self._cached_nl_description):
            return self._cached_nl_description

        # Build node summaries for prompt
        node_summaries = []
        for nid in path_nodes:
            node = self.nodes.get(nid, {})
            node_summaries.append(
                f"  - {node.get('name', nid)} (floor {node.get('level', '?')})"
            )

        prompt = f"""You are a navigation assistant for a visually impaired user.
Generate clear, spoken step-by-step directions following this path:

Path nodes in order:
{chr(10).join(node_summaries)}

Rules:
- Write in second person ("Turn left", "Walk forward")
- Mention floor changes explicitly ("Take the stairs up to floor 2")
- Keep each step under 15 words
- Include landmark cues from the node names
- End with "You have arrived at your destination"
- Output ONLY the numbered steps, nothing else"""

        try:
            response = client.chat.completions.create(
                model="qwen2.5:3b",
                max_tokens=400,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            description = response.choices[0].message.content.strip()
            self._cached_nl_description = description
            self._cached_for_start = start_node
            self._cached_for_goal = goal_node
            self._cached_path_nodes = path_nodes
            logger.info(f"[PathPlanner] NL description generated ({len(description)} chars)")
            return description
        except Exception as e:
            logger.error(f"[PathPlanner] LLM call failed: {e}")
            # Fallback: simple node name concatenation
            return " → ".join(
                self.nodes.get(n, {}).get("name", n) for n in path_nodes
            )

    # ── Step-by-step guidance ─────────────────────────────────────────────────

    def get_next_step_instruction(self, current_node: str, path_nodes: list[str]) -> str:
        """
        Returns the single next-step instruction for the current position.
        Prefers the VLM-generated edge description if available.

        Args:
            current_node: Node ID the user is currently at
            path_nodes:   Full planned path

        Returns:
            Short instruction string for next action
        """
        if not path_nodes:
            return "Continue following the planned route."

        if current_node not in path_nodes:
            return "Continue toward the next waypoint on your route."

        idx = path_nodes.index(current_node)
        if idx + 1 >= len(path_nodes):
            return "You have arrived at your destination."

        next_node = path_nodes[idx + 1]
        next_name = self.nodes.get(next_node, {}).get("name", next_node)
        curr_level = self.nodes.get(current_node, {}).get("level", 1)
        next_level = self.nodes.get(next_node, {}).get("level", 1)

        if next_level > curr_level:
            return f"Head to {next_name} and go up to floor {next_level}."
        elif next_level < curr_level:
            return f"Head to {next_name} and go down to floor {next_level}."
        else:
            return f"Proceed toward {next_name}."

    def get_edge_description(self, current_node: str, path_nodes: list[str]) -> str:
        """
        Returns the VLM-generated edge description between current node
        and the next node on the path. Falls back to empty string if
        no description was generated during scan walk.

        Args:
            current_node: Node ID the user is currently at
            path_nodes:   Full planned path

        Returns:
            Edge description string e.g. 'Turn left through the glass door.'
            or empty string if not available.
        """
        if not path_nodes or current_node not in path_nodes:
            return ""

        idx = path_nodes.index(current_node)
        if idx + 1 >= len(path_nodes):
            return ""

        next_node = path_nodes[idx + 1]
        return self.edges_map.get((current_node, next_node), "")
