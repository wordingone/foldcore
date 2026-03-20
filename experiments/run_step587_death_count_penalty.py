"""
Step 587 — Death-count penalty (ℓ_π candidate).

ℓ₁ (581d): penalty is FIXED (PENALTY=100 regardless of death history).
ℓ_π (this): penalty = death_count * UNIT — magnitude IS the substrate's experience.

Edge with 1 death → penalty UNIT.
Edge with 10 deaths → penalty 10*UNIT.
No prescribed constant — the data drives the magnitude.

Distinguishes: stochastic one-time accidents vs genuine death traps.
Naturally permanent (count only grows) — avoids 582 oscillation.

Three-way comparison on LS20, 5 seeds, 10K steps:
  A) DeathCount (UNIT=1)    — raw experience
  B) DeathCount (UNIT=10)   — scaled experience
  C) SoftPenalty (fixed=100) — 581d reference (ℓ₁)
  D) Argmin baseline

Key question: does B or C > D? And does B/C differ from 581d (ℓ₁)?
If death_count beats fixed penalty: substrate extracts information from repeated death.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 10_000
TIME_CAP = 60
PENALTY_FIXED = 100   # 581d's constant (ℓ₁ reference)
UNIT_1 = 1            # raw death_count
UNIT_10 = 10          # scaled death_count


# ── LSH hashing ──────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32)
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Death-count penalty (ℓ_π) ─────────────────────────────────────────────────

class DeathCountSub:
    """Penalty = death_count * unit. Magnitude is data-driven."""

    def __init__(self, lsh_seed=0, unit=1):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}                  # (node, action) -> {next_node: count}
        self.death_count = {}        # (node, action) -> death count
        self.unit = unit
        self._prev_node = None
        self._prev_action = None
        self.cells = set()
        self.total_deaths = 0
        self.penalty_applied_steps = 0

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

    def on_death(self):
        if self._prev_node is not None:
            key = (self._prev_node, self._prev_action)
            self.death_count[key] = self.death_count.get(key, 0) + 1
            self.total_deaths += 1

    def act(self):
        node = self._curr_node
        counts = np.array([sum(self.G.get((node, a), {}).values()) for a in range(N_A)],
                          dtype=np.float64)
        penalized = counts.copy()
        has_penalty = False
        for a in range(N_A):
            dc = self.death_count.get((node, a), 0)
            if dc > 0:
                penalized[a] += dc * self.unit
                has_penalty = True

        action = int(np.argmin(penalized))
        if has_penalty and action != int(np.argmin(counts)):
            self.penalty_applied_steps += 1

        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def diagnostics(self):
        death_edges = sum(1 for v in self.death_count.values() if v > 0)
        max_dc = max(self.death_count.values()) if self.death_count else 0
        return {
            'death_edges': death_edges,
            'max_death_count': max_dc,
            'total_deaths': self.total_deaths,
            'penalty_applied_steps': self.penalty_applied_steps,
        }


# ── Soft penalty (581d reference, ℓ₁) ────────────────────────────────────────

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
                penalized[a] += PENALTY_FIXED
        action = int(np.argmin(penalized))
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def on_death(self):
        if self._prev_node is not None:
            self.death_edges.add((self._prev_node, self._prev_action))
            self.total_deaths += 1


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

def run_seed(mk, seed, SubClass, label="", time_cap=TIME_CAP, **kwargs):
    env = mk()
    sub = SubClass(lsh_seed=seed * 100 + 7, **kwargs)
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
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    elapsed = time.time() - t0
    cells = len(sub.cells)
    deaths = getattr(sub, 'total_deaths', 0)
    print(f"  s{seed}: L1={l1} L2={l2} cells={cells} deaths={deaths} {elapsed:.0f}s",
          flush=True)
    if hasattr(sub, 'diagnostics'):
        d = sub.diagnostics()
        print(f"    diag: {d}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


def run_condition(mk, label, SubClass, n_seeds=5, **kwargs):
    print(f"\n--- {label} ---", flush=True)
    results = []
    t0 = time.time()
    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, SubClass, **kwargs)
        results.append(r)
    l1_total = sum(r['l1'] for r in results)
    l1_seeds = sum(1 for r in results if r['l1'] > 0)
    cells_avg = np.mean([r['cells'] for r in results])
    print(f"  {label}: {l1_seeds}/5 seeds L1, total={l1_total}, cells_avg={cells_avg:.0f}",
          flush=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print("Step 587: Death-count penalty (ℓ_π candidate)", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} UNIT_1={UNIT_1} UNIT_10={UNIT_10} "
          f"PENALTY_FIXED={PENALTY_FIXED}", flush=True)

    t_total = time.time()

    dc1  = run_condition(mk, "DeathCount(unit=1)",   DeathCountSub, unit=UNIT_1)
    dc10 = run_condition(mk, "DeathCount(unit=10)",  DeathCountSub, unit=UNIT_10)
    sp   = run_condition(mk, "SoftPenalty(fixed=100)", SoftPenaltySub)
    am   = run_condition(mk, "Argmin baseline",       ArgminSub)

    def summary(results):
        return (sum(r['l1'] for r in results),
                sum(1 for r in results if r['l1'] > 0),
                np.mean([r['cells'] for r in results]))

    dc1_l1, dc1_s, dc1_c   = summary(dc1)
    dc10_l1, dc10_s, dc10_c = summary(dc10)
    sp_l1,  sp_s,  sp_c    = summary(sp)
    am_l1,  am_s,  am_c    = summary(am)

    print(f"\n{'='*60}")
    print(f"Step 587: Death-count penalty vs fixed penalty vs argmin")
    print(f"  {'Condition':<26} {'Seeds':>6} {'L1':>4} {'Cells':>7}")
    print(f"  {'-'*50}")
    print(f"  {'DeathCount(unit=1)':<26} {dc1_s:>5}/5 {dc1_l1:>4} {dc1_c:>7.0f}")
    print(f"  {'DeathCount(unit=10)':<26} {dc10_s:>5}/5 {dc10_l1:>4} {dc10_c:>7.0f}")
    print(f"  {'SoftPenalty(fixed=100)':<26} {sp_s:>5}/5 {sp_l1:>4} {sp_c:>7.0f}")
    print(f"  {'Argmin':<26} {am_s:>5}/5 {am_l1:>4} {am_c:>7.0f}")

    # Verdict
    best_dc = max(dc1_l1, dc10_l1)
    best_dc_label = "DC(1)" if dc1_l1 >= dc10_l1 else "DC(10)"
    if best_dc > sp_l1:
        print(f"\n  ℓ_π SIGNAL: {best_dc_label}({best_dc}) > SoftPenalty({sp_l1}) > Argmin({am_l1})")
        print(f"  Death-count penalty extracts MORE from repeated death than fixed constant.")
    elif best_dc == sp_l1 and sp_l1 > am_l1:
        print(f"\n  ℓ_π NEUTRAL: DC({best_dc}) == SoftPenalty({sp_l1}) > Argmin({am_l1})")
        print(f"  Presence of penalty matters; magnitude does not (yet).")
    elif best_dc > am_l1:
        print(f"\n  ℓ_π PARTIAL: DC({best_dc}) > Argmin({am_l1}), but SP({sp_l1}) performs differently.")
    else:
        print(f"\n  ℓ_π FAIL: death-count penalty does not improve over argmin.")

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
