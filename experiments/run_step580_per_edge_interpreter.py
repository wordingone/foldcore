"""
Step 580 -- Per-edge interpreter selection.

Insight (Leo mail 2196): The frozen frame isn't argmin — it's the existence of
a SINGLE interpreter applied uniformly. What if different edges use different
selection rules, self-modified by interaction?

Setup: LSH k=12 + graph. Each edge stores:
  1. Visit count (as baseline)
  2. Selection flag: 0=argmin, 1=argmax, 2=random, 3=softmax

Action selection at node N:
  For each action A, read edge (N,A).flag → apply that rule to that edge's count.
  Combine per-edge results to choose action.

Self-modification: After each transition:
  - If action led to a NEW node → keep the edge's flag (exploration success)
  - If action led to an ALREADY-SEEN node → flip flag to a different rule

Edges near the frontier keep working rules. Interior edges rotate through
alternatives. The graph's selection policy becomes spatially heterogeneous.

R3 connection: edges learn which selection rule works for THEM.
Memory significance connection: moments that changed trajectory (new nodes)
preserve their interpreter; moments that didn't get modified.

5 seeds, 50K steps, LS20. Compare to pure argmin baseline.
Kill: 0/5 L1 → per-edge selection doesn't help.
Signal: L1 count > argmin baseline (3/5-5/5 at 50K).

Runtime cap: 5 min per seed.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 280  # seconds per seed
N_RULES = 4  # 0=argmin, 1=argmax, 2=random, 3=softmax
SOFTMAX_T = 1.0

# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    """64x64x1 frame → K-bit LSH hash."""
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Per-edge interpreter substrate ───────────────────────────────────────────

class PerEdgeSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        # Graph: (node, action) → {next_node: count}
        self.G = {}
        # Edge metadata: (node, action) → (visit_count, rule_flag)
        self.edge_meta = {}
        self._prev_node = None
        self._prev_action = None
        self.cells = set()
        self.rule_counts = np.zeros(N_RULES, dtype=np.int64)  # track which rules are active

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node

    def act(self):
        node = self._curr_node

        # Get edge info for each action
        counts = []
        flags = []
        for a in range(N_A):
            key = (node, a)
            if key in self.edge_meta:
                c, f = self.edge_meta[key]
            else:
                c, f = 0, 0  # default: 0 visits, argmin
            counts.append(c)
            flags.append(f)

        counts = np.array(counts, dtype=np.float64)
        flags = np.array(flags)

        # Per-edge selection: each edge votes using its own rule
        scores = np.zeros(N_A)
        for a in range(N_A):
            rule = flags[a]
            if rule == 0:  # argmin — prefer least visited
                scores[a] = -counts[a]
            elif rule == 1:  # argmax — prefer most visited
                scores[a] = counts[a]
            elif rule == 2:  # random
                scores[a] = np.random.random()
            elif rule == 3:  # softmax of inverse counts
                scores[a] = np.random.random()  # will be overridden below

        # For softmax edges: compute proper softmax over inverse counts
        softmax_mask = flags == 3
        if softmax_mask.any():
            inv = 1.0 / (counts[softmax_mask] + 1.0)
            inv = inv / SOFTMAX_T
            exp = np.exp(inv - inv.max())
            probs = exp / exp.sum()
            # Sample from softmax
            for i, a in enumerate(np.where(softmax_mask)[0]):
                scores[a] = probs[i] + np.random.random() * 0.001  # break ties

        action = int(np.argmax(scores))

        # Update edge visit count
        key = (node, action)
        if key in self.edge_meta:
            c, f = self.edge_meta[key]
            self.edge_meta[key] = (c + 1, f)
        else:
            self.edge_meta[key] = (1, 0)

        # Update transition graph
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

        # Self-modification: check if previous action led to new vs seen node
        if self._prev_node is not None:
            prev_key = (self._prev_node, self._prev_action)
            if prev_key in self.edge_meta:
                pc, pf = self.edge_meta[prev_key]
                # Was this node new when we arrived?
                # If the current node was first seen this step (only 1 visit from this edge)
                transitions = self.G.get(prev_key, {})
                times_arrived_here = transitions.get(node, 0)
                if times_arrived_here <= 1:
                    # NEW node reached — keep the rule (exploration success)
                    pass
                else:
                    # SEEN node — flip to a different rule
                    new_flag = (pf + 1) % N_RULES
                    self.edge_meta[prev_key] = (pc, new_flag)

        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def rule_distribution(self):
        """Count how many edges use each rule."""
        dist = np.zeros(N_RULES, dtype=np.int64)
        for (c, f) in self.edge_meta.values():
            dist[f] += 1
        return dist


# ── Argmin baseline ──────────────────────────────────────────────────────────

class ArgminSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node

    def act(self):
        node = self._curr_node
        counts = [sum(self.G.get((node, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None


# ── Seed runner ──────────────────────────────────────────────────────────────

def run_seed(mk, seed, SubClass, time_cap=TIME_CAP):
    env = mk()
    sub = SubClass(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    prev_cl = 0
    fresh = True
    l1 = l2 = go = step = 0
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < time_cap:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1

        if done:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 3:
                print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    elapsed = time.time() - t0
    cells = len(sub.cells)
    rule_dist = sub.rule_distribution() if hasattr(sub, 'rule_distribution') else None
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} {elapsed:.0f}s",
          flush=True)
    if rule_dist is not None:
        names = ['argmin', 'argmax', 'random', 'softmax']
        print(f"    rules: {dict(zip(names, rule_dist))}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print("Step 580: Per-edge interpreter selection", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} N_RULES={N_RULES}", flush=True)

    # Run per-edge substrate
    print("\n--- Per-edge interpreter ---", flush=True)
    pe_results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, PerEdgeSub)
        pe_results.append(r)

    # Run argmin baseline
    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    # Summary
    pe_l1 = sum(r['l1'] for r in pe_results)
    pe_seeds = sum(1 for r in pe_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 580: Per-edge interpreter selection")
    print(f"  Per-edge: {pe_seeds}/5 seeds L1, total L1={pe_l1}")
    for r in pe_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:   {am_seeds}/5 seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")

    if pe_l1 > am_l1:
        print(f"\nSIGNAL: per-edge ({pe_l1}) > argmin ({am_l1})")
    elif pe_l1 == am_l1:
        print(f"\nNEUTRAL: per-edge ({pe_l1}) == argmin ({am_l1})")
    else:
        print(f"\nFAIL: per-edge ({pe_l1}) < argmin ({am_l1})")


if __name__ == "__main__":
    main()
