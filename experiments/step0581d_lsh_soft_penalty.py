"""
Step 581d -- Soft penalty: death edges get visit count += PENALTY instead of hard block.

Argmin deprioritizes death edges heavily but doesn't forbid them. Full action space preserved.
Agent can still revisit death edges when all other edges are also heavily visited —
but death edges are at the back of the queue.

PENALTY=100: death edges appear to have 100 extra visits. Argmin picks them last.

Same budget: 5 seeds, 50K steps, LS20.
Kill: 0/5 OR cells < argmin baseline.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 280
PENALTY = 100  # virtual visits added to death edges

# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Soft penalty substrate ────────────────────────────────────────────────────

class SoftPenaltySub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}  # (node, action) -> {next_node: count}
        self.death_edges = set()  # edges that caused done=True (at least once)
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

        self.penalty_applied_steps = 0  # steps where penalty changed the choice
        self.total_deaths = 0

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

    def on_death(self):
        if self._prev_node is not None:
            self.death_edges.add((self._prev_node, self._prev_action))
            self.total_deaths += 1

    def act(self):
        node = self._curr_node
        counts = np.array([sum(self.G.get((node, a), {}).values()) for a in range(N_A)],
                          dtype=np.float64)

        # Apply penalty to death edges
        penalized = counts.copy()
        for a in range(N_A):
            if (node, a) in self.death_edges:
                penalized[a] += PENALTY

        action = int(np.argmin(penalized))

        # Track whether penalty changed the choice
        if action != int(np.argmin(counts)):
            self.penalty_applied_steps += 1

        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def diagnostics(self):
        return {
            'death_edges': len(self.death_edges),
            'total_deaths': self.total_deaths,
            'penalty_applied_steps': self.penalty_applied_steps,
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
        print(f"    death_edges={d['death_edges']} total_deaths={d['total_deaths']} "
              f"penalty_steps={d['penalty_applied_steps']}",
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

    print("Step 581d: Soft penalty death avoidance (PENALTY=100)", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY}", flush=True)

    print("\n--- SoftPenalty ---", flush=True)
    sp_results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, SoftPenaltySub)
        sp_results.append(r)

    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(5):
        if time.time() - t_total > 1380: print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    sp_l1 = sum(r['l1'] for r in sp_results)
    sp_seeds = sum(1 for r in sp_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 581d: Soft penalty death avoidance")
    print(f"  SoftPenalty: {sp_seeds}/5 seeds L1, total L1={sp_l1}")
    for r in sp_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:      {am_seeds}/5 seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")

    if sp_l1 > am_l1:
        print(f"\nSIGNAL: soft-penalty ({sp_l1}) > argmin ({am_l1})")
    elif sp_l1 == am_l1:
        print(f"\nNEUTRAL: soft-penalty ({sp_l1}) == argmin ({am_l1})")
    else:
        print(f"\nFAIL: soft-penalty ({sp_l1}) < argmin ({am_l1})")


if __name__ == "__main__":
    main()
