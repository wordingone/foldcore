"""
Step 581 -- Cerebellar dual-signal substrate: per-edge prediction error.

Each edge stores: visit count, predicted successor, prediction confidence.
Two signals:
  - Simple spike (default): argmin over visit counts
  - Complex spike (error): fires when actual != predicted successor

Action selection:
  - High confidence edges: follow predicted path (exploitation)
  - Low confidence edges: argmin explores them
  - Death edges learn to predict death → confidence rises → avoided

Key difference from Step 481 (0/10): per-edge prediction, not global frame.
Key difference from Step 580 (neutral): learns predictions, not fixed rule menu.

5 seeds, 50K steps, LS20. Compare to argmin baseline.
Kill: 0/5 L1 OR prediction confidence never rises above chance.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 280
CONFIDENCE_THRESHOLD = 3  # correct predictions before trusting an edge

# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Cerebellar substrate ─────────────────────────────────────────────────────

class CerebellarSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        # Transition graph: (node, action) → {next_node: count}
        self.G = {}
        # Per-edge predictions: (node, action) → {
        #   'predicted': predicted successor node (or None),
        #   'confidence': consecutive correct predictions,
        #   'visits': total visits
        # }
        self.edges = {}
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

        # Diagnostics
        self.correct_predictions = 0
        self.wrong_predictions = 0
        self.confident_follows = 0
        self.argmin_choices = 0

    def _get_edge(self, node, action):
        key = (node, action)
        if key not in self.edges:
            self.edges[key] = {
                'predicted': None,
                'confidence': 0,
                'visits': 0,
            }
        return self.edges[key]

    def observe(self, frame):
        node = encode(frame, self.H)
        is_new = node not in self.cells
        self.cells.add(node)
        self._curr_node = node

        # Update prediction for the previous edge
        if self._prev_node is not None:
            edge = self._get_edge(self._prev_node, self._prev_action)
            edge['visits'] += 1

            # Update transition graph
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

            # Prediction error check (complex spike)
            if edge['predicted'] is None:
                # First observation — set prediction, no error
                edge['predicted'] = node
                edge['confidence'] = 1
            elif edge['predicted'] == node:
                # Correct prediction — simple spike, increase confidence
                edge['confidence'] += 1
                self.correct_predictions += 1
            else:
                # Wrong prediction — complex spike, update prediction
                self.wrong_predictions += 1
                # Update to the MOST COMMON successor (not just the latest)
                transitions = self.G.get((self._prev_node, self._prev_action), {})
                if transitions:
                    edge['predicted'] = max(transitions, key=transitions.get)
                else:
                    edge['predicted'] = node
                edge['confidence'] = 0  # reset confidence

    def act(self):
        node = self._curr_node

        # Score each action: combine argmin exploration with prediction confidence
        counts = []
        confidences = []
        for a in range(N_A):
            edge = self._get_edge(node, a)
            counts.append(edge['visits'])
            confidences.append(edge['confidence'])

        counts = np.array(counts, dtype=np.float64)
        confidences = np.array(confidences, dtype=np.float64)

        # Strategy: if ANY edge has high confidence, prefer confident edges
        # (they lead to known, non-death outcomes). Among confident edges,
        # pick the least visited (explore the reliable paths).
        # If NO edge is confident, pure argmin (explore everything).
        confident_mask = confidences >= CONFIDENCE_THRESHOLD
        if confident_mask.any():
            # Among confident edges, pick least visited
            scores = np.where(confident_mask, -counts, 1e9)
            action = int(np.argmin(scores))
            self.confident_follows += 1
        else:
            # Pure argmin — explore
            action = int(np.argmin(counts))
            self.argmin_choices += 1

        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def diagnostics(self):
        total_edges = len(self.edges)
        confident_edges = sum(1 for e in self.edges.values()
                            if e['confidence'] >= CONFIDENCE_THRESHOLD)
        return {
            'total_edges': total_edges,
            'confident_edges': confident_edges,
            'correct_preds': self.correct_predictions,
            'wrong_preds': self.wrong_predictions,
            'confident_follows': self.confident_follows,
            'argmin_choices': self.argmin_choices,
        }


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

    prev_cl = 0; fresh = True
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
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} {elapsed:.0f}s",
          flush=True)
    if hasattr(sub, 'diagnostics'):
        d = sub.diagnostics()
        pred_rate = d['correct_preds'] / max(d['correct_preds'] + d['wrong_preds'], 1)
        print(f"    edges={d['total_edges']} confident={d['confident_edges']} "
              f"pred_acc={pred_rate:.3f} "
              f"confident_follows={d['confident_follows']} argmin={d['argmin_choices']}",
              flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print("Step 581: Cerebellar dual-signal substrate", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} CONF_THRESH={CONFIDENCE_THRESHOLD}", flush=True)

    print("\n--- Cerebellar ---", flush=True)
    cb_results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, CerebellarSub)
        cb_results.append(r)

    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    cb_l1 = sum(r['l1'] for r in cb_results)
    cb_seeds = sum(1 for r in cb_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 581: Cerebellar dual-signal")
    print(f"  Cerebellar: {cb_seeds}/5 seeds L1, total L1={cb_l1}")
    for r in cb_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:     {am_seeds}/5 seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")

    if cb_l1 > am_l1:
        print(f"\nSIGNAL: cerebellar ({cb_l1}) > argmin ({am_l1})")
    elif cb_l1 == am_l1:
        print(f"\nNEUTRAL: cerebellar ({cb_l1}) == argmin ({am_l1})")
    else:
        print(f"\nFAIL: cerebellar ({cb_l1}) < argmin ({am_l1})")


if __name__ == "__main__":
    main()
