from typing import List, Dict
from loguru import logger
from src.navigation.semantic_graph import SemanticGraph
import openai

class LLMRouteExplainer:
    """
    High-Level Semantic Route Summarizer.
    Calls LLM to generate human-centered multi-step instructions based on planned node sequence.
    """
    def __init__(self, graph: SemanticGraph, api_key: str = None, model: str = "gpt-4o-mini"):
        self.graph = graph
        self.model = model

        # If no API key provided, attempt to use local OpenAI compatible API (like Ollama)
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            logger.info("[LLM] No API key provided, defaulting to local Ollama on localhost:11434.")
            self.client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        self._cache = {}

    def summarize_route(self, path_nodes: List[str]) -> str:
        """
        Generates clear instructions for the user based on the path.
        """
        if not path_nodes:
            return "No route available."

        # Check cache
        path_tuple = tuple(path_nodes)
        if path_tuple in self._cache:
            return self._cache[path_tuple]

        # Build node context
        node_context = []
        for n_id in path_nodes:
            node = self.graph.get_node(n_id)
            if node:
                name = node.get("name", n_id)
                level = node.get("level", "?")
                area_type = node.get("areaType", "unknown")

                # Check if it's out of service (shouldn't happen if planner is good, but just in case)
                status = " (Warning: Marked out of service!)" if not node.get("status", {}).get("in_service", True) else ""

                node_context.append(f"- {name} (Floor: {level}, Type: {area_type}){status}")
            else:
                node_context.append(f"- {n_id}")

        context_str = "\n".join(node_context)

        prompt = f"""You are a helpful navigation assistant for a visually impaired user.
Based on the following sequence of locations they need to travel through, generate a clear, human-centered multi-step instruction summary.

Path sequence:
{context_str}

Important rules:
1. Be concise and conversational.
2. Mention any floor transitions (e.g., "take the stairs to the 15th floor").
3. Do not list the nodes directly; weave them into a natural sentence or two.
4. If there are out of service nodes bypassed (like an elevator), briefly mention favoring the stairs.

Example output: "Start at Room 1421, head towards the elevators which are out of service, take the stairs to the 15th floor, and proceed to Room 1521."
"""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.5
                )
                summary = response.choices[0].message.content.strip()
            else:
                # Fallback deterministic summary if no API client
                logger.warning("[LLM] Using fallback deterministic summarization.")
                summary = self._fallback_summarize(path_nodes)

            self._cache[path_tuple] = summary
            return summary

        except Exception as e:
            logger.error(f"[LLM] Route summarization failed: {e}")
            return self._fallback_summarize(path_nodes)

    def _fallback_summarize(self, path_nodes: List[str]) -> str:
        """Deterministic fallback if LLM is unavailable."""
        names = []
        for n_id in path_nodes:
            node = self.graph.get_node(n_id)
            if node:
                names.append(node.get("name", n_id))
            else:
                names.append(n_id)

        if len(names) <= 2:
            return f"Head directly from {names[0]} to {names[-1]}."

        return f"Start at {names[0]}, pass through several areas, and arrive at {names[-1]}."
