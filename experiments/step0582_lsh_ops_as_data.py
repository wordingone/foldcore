"""
Step 582 -- Operations as edge data. Argmin is the sole frozen interpreter.

Each edge stores: (visit_count, op_code, predicted_next)
op_codes:
  0 = count only (baseline)
  1 = count + predict (ontology only, not assigned by current rules)
  2 = count + PENALTY  (death avoidance — argmin deprioritizes)
  3 = count - BONUS    (prediction error boost — argmin prefers)

Self-modification rules (the substrate writes these, argmin reads them):
  death        -> op_code = 2
  succ != pred -> op_code = 3  (surprise: update prediction)
  succ == pred -> op_code = 0  (confirmed: nothing to learn)

Predictions update on every non-death traversal regardless of op_code.
Death overwrites op_code to 2, taking precedence over prediction error.

Action selection: argmin(effective_counts). Always. Two frozen ops:
  (1) read op_code, (2) apply to count. That's the entire frozen frame.

5 seeds, 50K steps, LS20. Compare to pure argmin (all op_code=0 forever).
Kill: 0/5 OR cells < argmin baseline.
Win:  >3/5 AND cells >= argmin.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 280
PENALTY = 100   # effective count bonus for death edges (op_code=2)
BONUS = 50      # effective count reduction for surprise edges (op_code=3)

# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Ops-as-data substrate ─────────────────────────────────────────────────────

class OpsDataSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}           # (node, action) -> {next_node: count}
        self.edge_op = {}     # (node, action) -> op_code (0/2/3)
        self.edge_pred = {}   # (node, action) -> predicted_next_node
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

        # Diagnostics
        self.total_deaths = 0
        self.op2_count = 0   # edges currently in op_code=2
        self.op3_count = 0   # edges currently in op_code=3
        self.surprise_events = 0

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node

        if self._prev_node is not None:
            key = (self._prev_node, self._prev_action)

            # Update visit count
            d = self.G.setdefault(key, {})
            d[node] = d.get(node, 0) + 1

            # Update prediction and op_code
            pred = self.edge_pred.get(key)
            prev_op = self.edge_op.get(key, 0)

            if pred is None:
                # First traversal: set prediction, default op
                self.edge_pred[key] = node
                self.edge_op[key] = 0
            elif pred == node:
                # Correct prediction: revert to default
                if prev_op != 0:
                    if prev_op == 2: self.op2_count -= 1
                    elif prev_op == 3: self.op3_count -= 1
                self.edge_op[key] = 0
            else:
                # Prediction error: surprise! update prediction, boost edge
                self.edge_pred[key] = node
                self.surprise_events += 1
                if prev_op != 3:
                    if prev_op == 2: self.op2_count -= 1
                    self.edge_op[key] = 3
                    self.op3_count += 1

    def on_death(self):
        if self._prev_node is not None:
            key = (self._prev_node, self._prev_action)
            prev_op = self.edge_op.get(key, 0)
            if prev_op != 2:
                if prev_op == 3: self.op3_count -= 1
                self.edge_op[key] = 2
                self.op2_count += 1
            self.total_deaths += 1

    def act(self):
        node = self._curr_node
        effective = []
        for a in range(N_A):
            key = (node, a)
            visit = sum(self.G.get(key, {}).values())
            op = self.edge_op.get(key, 0)
            if op == 2:
                effective.append(visit + PENALTY)
            elif op == 3:
                effective.append(max(0, visit - BONUS))
            else:
                effective.append(visit)

        action = int(np.argmin(effective))
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def diagnostics(self):
        total_edges = len(self.edge_op)
        op0 = sum(1 for v in self.edge_op.values() if v == 0)
        op2 = sum(1 for v in self.edge_op.values() if v == 2)
        op3 = sum(1 for v in self.edge_op.values() if v == 3)
        return {
            'total_edges': total_edges,
            'op0': op0, 'op2': op2, 'op3': op3,
            'total_deaths': self.total_deaths,
            'surprise_events': self.surprise_events,
        }


# ── Argmin baseline ───────────────────────────────────────────────────────────

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

    def on_death(self):
        pass


# ── Seed runner ───────────────────────────────────────────────────────────────

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
            sub.on_death()
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
        print(f"    edges={d['total_edges']} op0={d['op0']} op2={d['op2']} op3={d['op3']} "
              f"deaths={d['total_deaths']} surprises={d['surprise_events']}",
              flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print("Step 582: Operations as edge data", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY} BONUS={BONUS}", flush=True)

    print("\n--- OpsData ---", flush=True)
    od_results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, OpsDataSub)
        od_results.append(r)

    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    od_l1 = sum(r['l1'] for r in od_results)
    od_seeds = sum(1 for r in od_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 582: Operations as edge data")
    print(f"  OpsData: {od_seeds}/5 seeds L1, total L1={od_l1}")
    for r in od_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:  {am_seeds}/5 seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")

    if od_l1 > am_l1:
        print(f"\nSIGNAL: ops-data ({od_l1}) > argmin ({am_l1})")
    elif od_l1 == am_l1:
        print(f"\nNEUTRAL: ops-data ({od_l1}) == argmin ({am_l1})")
    else:
        print(f"\nFAIL: ops-data ({od_l1}) < argmin ({am_l1})")


if __name__ == "__main__":
    main()
