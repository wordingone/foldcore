"""
Step 553 — Which high-entropy graph edges are reducible?

Offline analysis of experiments/ls20_graph_dump.json (no env needed).

For each (node, action) with H > 0.5:
  REDUCIBLE: successors have distinct transition profiles (T profiles)
             -> parent mixes distinguishable states
  IRREDUCIBLE: successors have similar T profiles
               -> genuine stochasticity (death/reset)

Method: for each high-entropy edge, compute pairwise cosine distance
between successor transition fingerprints. High variance => REDUCIBLE.

Prediction: ~30% reducible, ~70% irreducible (noisy TV).
Kill: 0% reducible -> refinement pointless. >50% -> major opportunity.
"""
import json
import numpy as np
import sys
import os

H_THRESH = 0.5
REDUCIBLE_THRESH = 0.3  # max pairwise cosine distance to call it reducible


def load_graph(path):
    with open(path) as f:
        d = json.load(f)
    # Rebuild G: {(src, action): {dst: count}}
    G = {}
    for e in d['edges']:
        key = (e['src'], e['action'])
        G.setdefault(key, {})[e['dst']] = e['count']
    return d, G


def entropy(d):
    total = sum(d.values())
    if total < 2:
        return 0.0
    v = np.array(list(d.values()), np.float64)
    p = v / v.sum()
    return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))


def transition_fingerprint(G, node_id, n_actions=4):
    """Outgoing transition distribution for a node: sum over all actions."""
    fp = {}
    for a in range(n_actions):
        for dst, cnt in G.get((node_id, a), {}).items():
            fp[dst] = fp.get(dst, 0) + cnt
    return fp


def cosine_dist(fp_a, fp_b):
    """Cosine distance (1 - cosine_sim) between two fingerprint dicts."""
    all_keys = set(fp_a) | set(fp_b)
    if not all_keys:
        return 0.0
    va = np.array([fp_a.get(k, 0) for k in all_keys], np.float64)
    vb = np.array([fp_b.get(k, 0) for k in all_keys], np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(va, vb) / (na * nb))


def classify_edge(G, src, action, successors, min_count=3):
    """
    Return ('reducible'|'irreducible', max_pairwise_dist, groups).
    Uses pairwise cosine distance of successor T-profiles.
    """
    succ_fps = {}
    for s in successors:
        fp = transition_fingerprint(G, s)
        if fp:  # only consider successors with known outgoing edges
            succ_fps[s] = fp

    if len(succ_fps) < 2:
        return 'irreducible', 0.0, []

    # Pairwise distances
    succ_list = list(succ_fps.keys())
    max_dist = 0.0
    for i in range(len(succ_list)):
        for j in range(i + 1, len(succ_list)):
            d = cosine_dist(succ_fps[succ_list[i]], succ_fps[succ_list[j]])
            max_dist = max(max_dist, d)

    tag = 'reducible' if max_dist > REDUCIBLE_THRESH else 'irreducible'
    return tag, max_dist, succ_list


def t0():
    # entropy: uniform = 2 bits, singleton = 0
    assert abs(entropy({0: 5, 1: 5}) - 1.0) < 0.01, "uniform 2-way"
    assert abs(entropy({0: 10}) - 0.0) < 0.01, "singleton"
    assert abs(entropy({0: 5, 1: 5, 2: 5, 3: 5}) - 2.0) < 0.01, "uniform 4-way"

    # cosine_dist: identical = 0, orthogonal = 1
    fp1 = {"a": 3, "b": 2}
    assert abs(cosine_dist(fp1, fp1) - 0.0) < 1e-6
    assert abs(cosine_dist({"a": 1}, {"b": 1}) - 1.0) < 1e-6

    # partial overlap
    d = cosine_dist({"a": 1, "b": 1}, {"a": 1, "c": 1})
    assert 0.3 < d < 0.7, f"partial overlap dist={d:.3f}"

    # REDUCIBLE: successors with opposite fingerprints
    G_test = {
        ("s1", 0): {"x": 10, "y": 0},   # s1 goes to x
        ("s2", 0): {"x": 0, "y": 10},   # s2 goes to y (opposite)
    }
    tag, max_d, _ = classify_edge(G_test, "n", 0, ["s1", "s2"])
    assert tag == 'reducible', f"Expected reducible, got {tag} (max_d={max_d:.3f})"

    # IRREDUCIBLE: successors with identical fingerprints
    G_test2 = {
        ("s1", 0): {"x": 5, "y": 5},
        ("s2", 0): {"x": 5, "y": 5},
    }
    tag2, max_d2, _ = classify_edge(G_test2, "n", 0, ["s1", "s2"])
    assert tag2 == 'irreducible', f"Expected irreducible, got {tag2}"

    print("T0 PASS")


def main():
    t0()

    graph_path = "experiments/ls20_graph_dump.json"
    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        return

    data, G = load_graph(graph_path)
    meta = data.get('meta', {})
    print(f"Graph: {meta.get('live_nodes')} live + {meta.get('refined_nodes')} refined "
          f"= {meta.get('total_nodes_ever')} total nodes, "
          f"{meta.get('total_edges')} edges", flush=True)

    # Find high-entropy edges
    high_entropy_edges = []
    for (src, action), d in G.items():
        h = entropy(d)
        if h > H_THRESH:
            high_entropy_edges.append((h, src, action, d))

    high_entropy_edges.sort(key=lambda x: x[0], reverse=True)
    print(f"High-entropy edges (H > {H_THRESH}): {len(high_entropy_edges)}", flush=True)

    if not high_entropy_edges:
        print("No high-entropy edges found.")
        return

    # Classify each
    results = []
    for h, src, action, d in high_entropy_edges:
        successors = list(d.keys())
        tag, max_d, groups = classify_edge(G, src, action, successors)
        results.append(dict(h=h, src=src, action=action, tag=tag,
                            max_dist=max_d, n_succ=len(successors),
                            obs=sum(d.values())))

    n_reducible = sum(1 for r in results if r['tag'] == 'reducible')
    n_irreducible = sum(1 for r in results if r['tag'] == 'irreducible')
    pct = n_reducible / len(results) if results else 0.0

    print(f"\nResults: {n_reducible} reducible / {n_irreducible} irreducible "
          f"({pct:.1%} reducible)")
    print(f"Threshold: cosine_dist > {REDUCIBLE_THRESH} => reducible")

    # Top reducible
    reducible = sorted([r for r in results if r['tag'] == 'reducible'],
                       key=lambda x: x['h'], reverse=True)
    print(f"\nTop 10 REDUCIBLE (highest entropy, exploitable structure):")
    for r in reducible[:10]:
        print(f"  node={r['src']} action={r['action']} H={r['h']:.3f} "
              f"obs={r['obs']} succ={r['n_succ']} max_dist={r['max_dist']:.3f}")

    # Top irreducible
    irreducible = sorted([r for r in results if r['tag'] == 'irreducible'],
                         key=lambda x: x['h'], reverse=True)
    print(f"\nTop 10 IRREDUCIBLE (genuine noise/death transitions):")
    for r in irreducible[:10]:
        print(f"  node={r['src']} action={r['action']} H={r['h']:.3f} "
              f"obs={r['obs']} succ={r['n_succ']} max_dist={r['max_dist']:.3f}")

    # Distribution of max_dist
    all_dists = [r['max_dist'] for r in results]
    print(f"\nmax_dist distribution (proxy for reducibility):")
    print(f"  mean={np.mean(all_dists):.3f} median={np.median(all_dists):.3f} "
          f"max={max(all_dists):.3f}")
    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        count = sum(1 for d in all_dists if d > thresh)
        print(f"  > {thresh}: {count}/{len(all_dists)} ({count/len(all_dists):.1%})")

    print(f"\n{'='*60}")
    if pct == 0.0:
        print("KILL: 0% reducible. All confusion is noise. Refinement pointless post-saturation.")
    elif pct > 0.50:
        print(f"OPPORTUNITY: {pct:.1%} reducible. Major refinement opportunity. "
              "L2 path may exist through better splitting.")
    else:
        print(f"PARTIAL: {pct:.1%} reducible. Some exploitable structure, "
              "mostly genuine noise.")


if __name__ == "__main__":
    main()
