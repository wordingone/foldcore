"""
Step 584 — 581d seed expansion: 20 seeds on LS20 (FULL MODE, 50K steps).

Approved. Checkpoints at 10K/20K/30K/40K/50K.
Early kill: if deaths=0 across ALL seeds at 20K, abort — mechanism never fires.

Output: X/20 seeds L1 per checkpoint, Fisher exact p-value at 50K.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 300    # 5 min per seed
N_SEEDS = 20
PENALTY = 100
CHECKPOINTS = [10_000, 20_000, 30_000, 40_000, 50_000]
EARLY_KILL_STEP = 20_000  # if deaths=0 for ALL seeds at this point, abort


# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Soft penalty substrate (identical to 581d) ────────────────────────────────

class SoftPenaltySub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self._prev_node = None
        self._prev_action = None
        self.cells = set()
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
        penalized = counts.copy()
        for a in range(N_A):
            if (node, a) in self.death_edges:
                penalized[a] += PENALTY
        action = int(np.argmin(penalized))
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None


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


# ── Seed runner with checkpoints ─────────────────────────────────────────────

def run_seed(mk, seed, SubClass, time_cap=TIME_CAP):
    env = mk()
    sub = SubClass(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    prev_cl = 0; fresh = True
    l1 = l2 = go = step = 0
    t0 = time.time()
    checkpoints = {}       # step -> {l1, cells, deaths}
    next_ckpt_idx = 0

    while step < MAX_STEPS and time.time() - t0 < time_cap:
        # Checkpoint
        while next_ckpt_idx < len(CHECKPOINTS) and step >= CHECKPOINTS[next_ckpt_idx]:
            ck = CHECKPOINTS[next_ckpt_idx]
            cells = len(sub.cells)
            deaths = getattr(sub, 'total_deaths', 0)
            checkpoints[ck] = dict(l1=l1, cells=cells, deaths=deaths)
            print(f"    CK@{ck//1000}K: L1={l1} cells={cells} deaths={deaths}", flush=True)
            next_ckpt_idx += 1

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
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    # Final checkpoint flush
    while next_ckpt_idx < len(CHECKPOINTS) and step >= CHECKPOINTS[next_ckpt_idx]:
        ck = CHECKPOINTS[next_ckpt_idx]
        cells = len(sub.cells)
        deaths = getattr(sub, 'total_deaths', 0)
        checkpoints[ck] = dict(l1=l1, cells=cells, deaths=deaths)
        next_ckpt_idx += 1

    elapsed = time.time() - t0
    cells = len(sub.cells)
    deaths = getattr(sub, 'total_deaths', 0)
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} deaths={deaths} {elapsed:.0f}s",
          flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells,
                deaths=deaths, checkpoints=checkpoints)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from scipy.stats import fisher_exact
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("WARNING: scipy not available, skipping Fisher test", flush=True)

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 584: Seed expansion FULL — {N_SEEDS} seeds, LS20, 50K steps", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY} TIME_CAP={TIME_CAP}s/seed", flush=True)
    print(f"  Checkpoints: {CHECKPOINTS}", flush=True)
    print(f"  Early kill: if deaths=0 for ALL seeds at {EARLY_KILL_STEP} steps", flush=True)

    sp_results = []
    t_total = time.time()

    print("\n--- SoftPenalty (581d config) ---", flush=True)
    for seed in range(N_SEEDS):
        elapsed_total = time.time() - t_total
        print(f"\nseed {seed} (total {elapsed_total:.0f}s):", flush=True)
        r = run_seed(mk, seed, SoftPenaltySub)
        sp_results.append(r)

        # Early kill check after all seeds reach EARLY_KILL_STEP
        if seed == N_SEEDS - 1:
            ek_deaths = sum(r2.get('checkpoints', {}).get(EARLY_KILL_STEP, {}).get('deaths', 0)
                            for r2 in sp_results)
            if ek_deaths == 0:
                print(f"\n  EARLY KILL: deaths=0 for all {N_SEEDS} seeds at {EARLY_KILL_STEP} steps.", flush=True)
                print(f"  Death penalty mechanism never fired. Aborting — 50K would not change this.", flush=True)
                # Print partial results and exit
                sp_l1 = sum(r2['l1'] for r2 in sp_results)
                sp_seeds = sum(1 for r2 in sp_results if r2['l1'] > 0)
                print(f"\n  Partial SoftPenalty: {sp_seeds}/{N_SEEDS} seeds L1, total L1={sp_l1}")
                print(f"  (ABORTED: deaths=0, mechanism inactive)")
                return

    print("\n--- Argmin baseline ---", flush=True)
    am_results = []
    for seed in range(N_SEEDS):
        elapsed_total = time.time() - t_total
        print(f"\nseed {seed} (total {elapsed_total:.0f}s):", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    sp_l1 = sum(r['l1'] for r in sp_results)
    sp_seeds = sum(1 for r in sp_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print(f"Step 584: Seed expansion FULL ({N_SEEDS} seeds, 50K steps)")
    print(f"  SoftPenalty: {sp_seeds}/{N_SEEDS} seeds L1, total L1={sp_l1}")
    for r in sp_results:
        print(f"    s{r['seed']:02d}: L1={r['l1']} cells={r['cells']} deaths={r.get('deaths',0)}")
    print(f"  Argmin:      {am_seeds}/{N_SEEDS} seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']:02d}: L1={r['l1']} cells={r['cells']} deaths={r.get('deaths',0)}")

    # Checkpoint table: SP L1 vs AM L1 at each step milestone
    print(f"\n  Checkpoint progression (SP_L1 / AM_L1 across seeds):")
    for ck in CHECKPOINTS:
        sp_ck_l1 = sum(r.get('checkpoints', {}).get(ck, {}).get('l1', 0) for r in sp_results)
        am_ck_l1 = sum(r.get('checkpoints', {}).get(ck, {}).get('l1', 0) for r in am_results)
        sp_ck_d  = sum(r.get('checkpoints', {}).get(ck, {}).get('deaths', 0) for r in sp_results)
        am_ck_d  = sum(r.get('checkpoints', {}).get(ck, {}).get('deaths', 0) for r in am_results)
        print(f"    @{ck//1000:2d}K: SP_L1={sp_ck_l1:3d} AM_L1={am_ck_l1:3d} "
              f"SP_deaths={sp_ck_d:4d} AM_deaths={am_ck_d:4d}")

    # Fisher exact test: proportion of seeds reaching L1
    if HAS_SCIPY:
        # 2x2 table: [sp_l1_seeds, sp_not] vs [am_l1_seeds, am_not]
        table = [[sp_seeds, N_SEEDS - sp_seeds],
                 [am_seeds, N_SEEDS - am_seeds]]
        odds, pval = fisher_exact(table, alternative='greater')
        print(f"\n  Fisher exact (SP > argmin): odds={odds:.3f} p={pval:.4f}")
        if pval < 0.05:
            print(f"  SIGNIFICANT: p={pval:.4f} < 0.05")
        elif pval < 0.10:
            print(f"  TREND: p={pval:.4f} < 0.10")
        else:
            print(f"  NOT SIGNIFICANT: p={pval:.4f}")
    else:
        print(f"\n  SoftPenalty: {sp_seeds}/{N_SEEDS}, Argmin: {am_seeds}/{N_SEEDS}")
        print(f"  (install scipy for Fisher exact test)")

    total_elapsed = time.time() - t_total
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
