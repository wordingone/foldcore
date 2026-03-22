"""
Step 563: Q4 — What fraction of refinements actually expand the frontier?

Load graph dump (Step 550). For each refined parent: do BFS from children and compare
to BFS from parent. Useful refinement = children reach nodes parent couldn't.

Prediction: <5% useful. Most refinements are redundant after 500K steps.
Offline analysis. Fast.
"""
import json
import numpy as np
from collections import defaultdict, deque


def load_graph(path):
    with open(path) as f:
        d = json.load(f)
    return d


def build_adj(edges):
    """Build adjacency list: node -> set of neighbors (any action)."""
    adj = defaultdict(set)
    for e in edges:
        adj[e['src']].add(e['dst'])
    return adj


def bfs_reachable(start, adj):
    """BFS from start, return set of reachable nodes."""
    if start not in adj and not any(start in s for s in adj.values()):
        return {start}  # isolated
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


def t0():
    # Simple BFS test
    adj = {'A': {'B', 'C'}, 'B': {'D'}, 'C': {'D'}, 'D': set()}
    r = bfs_reachable('A', adj)
    assert r == {'A', 'B', 'C', 'D'}
    r2 = bfs_reachable('D', adj)
    assert r2 == {'D'}
    print("T0 PASS")


def main():
    t0()

    d = load_graph('ls20_graph_dump.json')
    print(f"Graph: {d['meta']}")

    nodes = {n['id']: n for n in d['nodes']}
    refined_ids = [n['id'] for n in d['nodes'] if n['type'] == 'refined']
    print(f"Refined nodes: {len(refined_ids)}")

    adj = build_adj(d['edges'])
    node_id_set = set(nodes.keys())

    # For each refined node, find its children
    results = []
    no_children = 0

    for rid in refined_ids:
        c0 = f'({rid}, 0)'
        c1 = f'({rid}, 1)'
        children = [c for c in (c0, c1) if c in node_id_set]
        if not children:
            no_children += 1
            continue

        # BFS from parent
        r_parent = bfs_reachable(rid, adj)

        # BFS from children (union)
        r_children = set()
        for ch in children:
            r_children |= bfs_reachable(ch, adj)

        # New nodes only reachable from children
        new_nodes = r_children - r_parent

        # Get timing: use min first_visit of children as proxy for when refinement occurred
        child_visits = [int(nodes[c]['first_visit']) for c in children if c in nodes]
        refine_time = min(child_visits) if child_visits else 0

        results.append({
            'parent': rid,
            'children': children,
            'r_parent': len(r_parent),
            'r_children': len(r_children),
            'new_nodes': len(new_nodes),
            'useful': len(new_nodes) > 0,
            'refine_time': refine_time
        })

    print(f"Analyzed: {len(results)} refinements ({no_children} had no children in dump)")

    useful = [r for r in results if r['useful']]
    redundant = [r for r in results if not r['useful']]
    rate = len(useful) / len(results) if results else 0.0

    print(f"\n=== Refinement utility ===")
    print(f"Total analyzed: {len(results)}")
    print(f"Useful (expand frontier): {len(useful)} ({rate:.1%})")
    print(f"Redundant (no new nodes): {len(redundant)} ({1-rate:.1%})")

    if useful:
        avg_new = np.mean([r['new_nodes'] for r in useful])
        print(f"Avg new nodes per useful refinement: {avg_new:.1f}")

    # Decay analysis: useful_rate by time quartile
    results.sort(key=lambda r: r['refine_time'])
    n = len(results)
    if n >= 4:
        print(f"\n=== Useful rate by time quartile ===")
        for q in range(4):
            lo = q * n // 4
            hi = (q + 1) * n // 4
            chunk = results[lo:hi]
            u = sum(1 for r in chunk if r['useful'])
            t_lo = chunk[0]['refine_time'] if chunk else 0
            t_hi = chunk[-1]['refine_time'] if chunk else 0
            print(f"  Q{q+1} (steps {t_lo}-{t_hi}): {u}/{len(chunk)} useful ({u/len(chunk):.1%})")

    print(f"\n=== Summary ===")
    print(f"Useful refinement rate: {rate:.1%}")
    print(f"Baseline (Step 549): Jaccard 0.951->0.798 = refinements DON'T expand frontier")
    if rate < 0.05:
        print(f"CONFIRMED: <5% useful. Refinements are mostly redundant.")
    elif rate < 0.20:
        print(f"Low utility ({rate:.1%}). Most refinements redundant.")
    else:
        print(f"Higher utility than expected ({rate:.1%}). Some refinements create new paths.")


if __name__ == "__main__":
    main()
