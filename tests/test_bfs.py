"""
tests/test_bfs.py

Test 4: BFS path planner on an OSMAG JSON file.

Usage:
    python tests/test_bfs.py --osmag maps/<building>/osmag.json

Expected: Prints path between first and last node, tests goal resolution.
"""

import argparse
import sys
import json

sys.path.insert(0, ".")

from modules.m4_path_planner import PathPlanner

parser = argparse.ArgumentParser(description="Test BFS path planner on OSMAG graph.")
parser.add_argument("--osmag", required=True, help="Path to osmag.json")
parser.add_argument("--from-node", help="Start node ID (optional, defaults to first node)")
parser.add_argument("--to-node",   help="Goal node ID (optional, defaults to last node)")
parser.add_argument("--goal-word", help="Test goal resolution with this spoken word")
args = parser.parse_args()

print(f"=== BFS Path Planner Test ===")
print(f"OSMAG: {args.osmag}")

# Load planner
planner = PathPlanner(args.osmag)
nodes = list(planner.nodes.keys())
print(f"Nodes: {len(nodes)}")
print(f"Edges: {sum(len(v) for v in planner.graph.values()) // 2} (bidirectional)")

# Print node list
print("\nAll nodes:")
for nid, node in planner.nodes.items():
    lm = ", ".join(node.get("landmarks", [])[:2])
    print(f"  [{nid}] {node['name']}  (floor {node.get('level', 1)})  {f'  landmarks: {lm}' if lm else ''}")

# BFS path test
start = args.from_node or nodes[0]
goal  = args.to_node   or nodes[-1]

print(f"\nTest 1: BFS  {start}  →  {goal}")
path = planner.find_path(start, goal)
if path:
    print(f"  Path ({len(path)} nodes):")
    for nid in path:
        print(f"    → {nid}: {planner.nodes.get(nid, {}).get('name', nid)}")
else:
    print("  WARN: No path found (graph may not be connected)")

# Next-step instructions
if path and len(path) >= 2:
    print("\nTest 2: Next-step instructions along path")
    for nid in path:
        instr = planner.get_next_step_instruction(nid, path)
        print(f"  At [{nid}]: {instr}")

# Goal resolution test
if args.goal_word:
    print(f"\nTest 3: Goal resolution for '{args.goal_word}'")
    resolved = planner.resolve_goal_node(args.goal_word)
    if resolved:
        print(f"  Resolved → [{resolved}]: {planner.nodes[resolved]['name']}")
    else:
        print(f"  WARN: Could not resolve '{args.goal_word}' to any node")
else:
    # Test common words automatically
    print("\nTest 3: Goal resolution for common words")
    test_words = ["restroom", "stairs", "elevator", "entrance", "room", "corridor"]
    for word in test_words:
        r = planner.resolve_goal_node(word)
        status = f"→ [{r}] {planner.nodes[r]['name']}" if r else "no match"
        print(f"  '{word}': {status}")

# Same-node path
print(f"\nTest 4: Same-node path ({start} → {start})")
same_path = planner.find_path(start, start)
assert same_path == [start], f"Expected [{start}], got {same_path}"
print(f"  Correctly returned [{start}]")

print("\n✓ BFS test PASSED")
