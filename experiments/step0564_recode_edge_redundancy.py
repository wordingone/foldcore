"""
Step 564: Q5 — What fraction of high-traffic edges are irredundant (load-bearing)?

For each edge with >50 observations: is it a bridge? (removing it disconnects n' from n)
Bridge = irredundant = the only path from n to n' in the directed graph.

Prediction: <20% irredundant. Most edges are redundant (parallel paths exist).
Offline analysis of graph dump. Fast.
"""
import json
import numpy as np
from collections import defaultdict, deque


def load_graph(path):
    with open(path) as f:
        return json.load(f)


def build_adj(edges, exclude=None):
    """Build adjacency list. Optionally exclude one edge (src, dst)."""
    adj = defaultdict(set)
    for e in edges:
        if exclude and (e['src'], e['dst']) == exclude:
            continue
        adj[e['src']].add(e['dst'])
    return adj


def reachable_from(start, adj):
    """BFS: return True if dst is reachable from start."""
    visited = set()
    q = deque([start])
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        for nb in adj.get(n, set()):
            if nb not in visited:
                q.append(nb)
    return visited


def is_bridge(src, dst, edges_by_src, all_edges):
    """Is the edge src->dst a bridge? (only path from src to dst)"""
    # Build adj without this edge
    adj = defaultdict(set)
    for e in all_edges:
        if e['src'] == src and e['dst'] == dst:
            continue  # exclude this edge
        adj[e['src']].add(e['dst'])
    # Check if dst still reachable from src
    visited = reachable_from(src, adj)
    return dst not in visited


def t0():
    # Graph: A->B (non-bridge, parallel via A->C->B), B->D (bridge, only path)
    edges = [
        {'src': 'A', 'dst': 'B', 'count': '100'},  # direct A->B
        {'src': 'A', 'dst': 'C', 'count': '100'},  # A->C->B (parallel)
        {'src': 'C', 'dst': 'B', 'count': '100'},
        {'src': 'B', 'dst': 'D', 'count': '100'},  # B->D: only path to D
    ]
    # A->B is NOT a bridge: remove it, A can still reach B via A->C->B
    assert not is_bridge('A', 'B', {}, edges)
    # B->D IS a bridge: remove it, D is unreachable from B
    assert is_bridge('B', 'D', {}, edges)
    print("T0 PASS")


def main():
    t0()

    d = load_graph('ls20_graph_dump.json')
    print(f"Graph: {d['meta']}")

    all_edges = d['edges']
    heavy_edges = [e for e in all_edges if int(e['count']) > 50]
    print(f"Heavy edges (count>50): {len(heavy_edges)} / {len(all_edges)} total")

    # Build per-src adjacency for quick lookups
    edges_by_src = defaultdict(list)
    for e in all_edges:
        edges_by_src[e['src']].append(e)

    # Also build multi-edge structure: (src, dst) -> count (aggregated)
    edge_map = defaultdict(int)
    for e in all_edges:
        edge_map[(e['src'], e['dst'])] += int(e['count'])

    # For efficiency: use global adj for BFS, modify per-test
    # Build a base adj without the heavy edge, recheck
    results = []
    t0_time = __import__('time').time()

    for e in heavy_edges:
        src, dst = e['src'], e['dst']
        # Check if this specific directed edge is the only path src->dst
        bridge = is_bridge(src, dst, edges_by_src, all_edges)
        results.append({
            'src': src, 'dst': dst,
            'count': int(e['count']),
            'bridge': bridge
        })

        if __import__('time').time() - t0_time > 50:
            print(f"  Timeout at {len(results)}/{len(heavy_edges)} edges")
            break

    if not results:
        print("No results.")
        return

    irredundant = [r for r in results if r['bridge']]
    redundant = [r for r in results if not r['bridge']]
    rate = len(irredundant) / len(results)

    print(f"\n=== Edge redundancy analysis ===")
    print(f"Analyzed: {len(results)} heavy edges")
    print(f"Irredundant (bridges): {len(irredundant)} ({rate:.1%})")
    print(f"Redundant (parallel paths exist): {len(redundant)} ({1-rate:.1%})")

    if irredundant:
        avg_count = np.mean([r['count'] for r in irredundant])
        print(f"Avg traffic on irredundant edges: {avg_count:.0f}")
    if redundant:
        avg_count = np.mean([r['count'] for r in redundant])
        print(f"Avg traffic on redundant edges: {avg_count:.0f}")

    # Top irredundant edges by count
    irredundant.sort(key=lambda r: -r['count'])
    print(f"\nTop 5 irredundant edges (highest traffic):")
    for r in irredundant[:5]:
        print(f"  {r['src']} -> {r['dst']}: count={r['count']}")

    print(f"\n=== Summary ===")
    print(f"Irredundant rate: {rate:.1%}")
    print(f"Prediction: <20%")
    if rate < 0.20:
        print(f"CONFIRMED: <20% irredundant. Graph has many parallel paths.")
        print(f"Navigation robust: most edges can be lost without disconnecting the graph.")
    else:
        print(f"Higher than expected ({rate:.1%}). Graph is less redundant than predicted.")


if __name__ == "__main__":
    main()
