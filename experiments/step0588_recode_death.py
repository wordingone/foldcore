"""
Step 588 -- Recode + SoftPenalty combination on LS20.

Research question: does Recode (expansion) + death penalty (avoidance) beat either alone?
Recode expands the reachable set via adaptive node splitting (K=16).
SoftPenalty accelerates traversal by marking death edges (K=12).
Combined: Recode with death edges.

Four conditions, 5 seeds, 10K steps each:
  A) Recode alone      (K=16, adaptive splits, no death penalty) -- step 542 baseline
  B) Recode+Death      (K=16, adaptive splits + PENALTY=100 on death edges)
  C) SoftPenalty alone (K=12, fixed penalty, no splits) -- 581d baseline
  D) Argmin            (K=12, no penalty, no splits) -- baseline
"""
import numpy as np
import time
import sys

# Shared config
N_A = 4
PENALTY = 100      # death penalty constant (581d value)

# Recode config (step 542)
K_RECODE = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05

# SoftPenalty config (581d)
K_SP = 12

MAX_STEPS = 10_000
TIME_CAP = 60      # seconds per seed


# -- Encoding -----------------------------------------------------------

def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()

def lsh_hash(x, H):
    """Simple LSH hash (for SoftPenalty / Argmin)."""
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


# -- Recode substrate (from step 542) ------------------------------------

class Recode:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K_RECODE, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self.dim = DIM

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_vec(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + x.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        pass

    def on_reset(self):
        self._pn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            did += 1
            if did >= 3:
                break

    def stats(self):
        return len(self.live), len(self.ref), len(self.G)


# -- Recode + Death penalty ---------------------------------------------

class RecodeDeath(Recode):
    """Recode with permanent soft death penalty on death edges."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.death_edges = set()
        self.total_deaths = 0

    def on_death(self):
        if self._pn is not None:
            self.death_edges.add((self._pn, self._pa))
            self.total_deaths += 1

    def act(self):
        counts = np.array(
            [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)],
            dtype=np.float64
        )
        for a in range(N_A):
            if (self._cn, a) in self.death_edges:
                counts[a] += PENALTY
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action


# -- SoftPenalty (581d, K=12) -------------------------------------------

class SoftPenalty:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K_SP, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame)
        n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = np.array(
            [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)],
            dtype=np.float64
        )
        for a in range(N_A):
            if (self._cn, a) in self.death_edges:
                counts[a] += PENALTY
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        if self._pn is not None:
            self.death_edges.add((self._pn, self._pa))
            self.total_deaths += 1

    def on_reset(self):
        self._pn = None


# -- Argmin (K=12) ------------------------------------------------------

class Argmin:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K_SP, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame)
        n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        pass

    def on_reset(self):
        self._pn = None


# -- Seed runner --------------------------------------------------------

def run_seed(mk, seed, SubClass):
    env = mk()
    sub = SubClass(seed=seed * 1000 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = l2 = go = step = 0
    prev_cl = 0; fresh = True
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
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
    deaths = getattr(sub, 'total_deaths', 0)
    if hasattr(sub, 'stats'):
        nc, ns, ne = sub.stats()
        cells = nc
    else:
        cells = len(getattr(sub, 'cells', set()))
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} "
          f"deaths={deaths} {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells, deaths=deaths)


def run_condition(mk, label, SubClass, n_seeds=5):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        try:
            r = run_seed(mk, seed, SubClass)
            results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)
    l1 = sum(r['l1'] for r in results)
    seeds = sum(1 for r in results if r['l1'] > 0)
    avg_cells = np.mean([r['cells'] for r in results]) if results else 0
    print(f"  {label}: {seeds}/{n_seeds} L1={l1} cells_avg={avg_cells:.0f}", flush=True)
    return results


# -- Main ---------------------------------------------------------------

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print("Step 588: Recode + SoftPenalty combination on LS20", flush=True)
    print(f"  5 seeds x {MAX_STEPS} steps | PENALTY={PENALTY} | K_Recode={K_RECODE} K_SP={K_SP}",
          flush=True)

    t_total = time.time()

    rc   = run_condition(mk, "Recode alone (K=16)",       Recode)
    rcd  = run_condition(mk, "Recode+Death (K=16)",       RecodeDeath)
    sp   = run_condition(mk, "SoftPenalty alone (K=12)",  SoftPenalty)
    am   = run_condition(mk, "Argmin (K=12)",              Argmin)

    def summarize(results):
        l1 = sum(r['l1'] for r in results)
        seeds = sum(1 for r in results if r['l1'] > 0)
        return seeds, l1

    rc_s, rc_l1   = summarize(rc)
    rcd_s, rcd_l1 = summarize(rcd)
    sp_s, sp_l1   = summarize(sp)
    am_s, am_l1   = summarize(am)

    print(f"\n{'='*60}")
    print(f"Step 588: Recode + SoftPenalty combination")
    print(f"  {'Condition':<28} {'Seeds':>6} {'L1':>4}")
    print(f"  {'-'*40}")
    print(f"  {'Recode alone (K=16)':<28} {rc_s:>5}/5 {rc_l1:>4}")
    print(f"  {'Recode+Death (K=16)':<28} {rcd_s:>5}/5 {rcd_l1:>4}")
    print(f"  {'SoftPenalty (K=12)':<28} {sp_s:>5}/5 {sp_l1:>4}")
    print(f"  {'Argmin (K=12)':<28} {am_s:>5}/5 {am_l1:>4}")

    # Verdict
    print()
    if rcd_l1 > rc_l1 and rcd_l1 > sp_l1:
        print(f"  ADDITIVE SIGNAL: Recode+Death({rcd_l1}) > Recode({rc_l1}) and SP({sp_l1})")
        print(f"  Expansion + avoidance are complementary.")
    elif rcd_l1 > rc_l1 and rcd_l1 == sp_l1:
        print(f"  PARTIAL: Recode+Death({rcd_l1}) > Recode({rc_l1}), ties SP({sp_l1}).")
        print(f"  Death penalty helps Recode but doesn't exceed SP.")
    elif rcd_l1 == rc_l1 and rcd_l1 >= sp_l1:
        print(f"  NEUTRAL: Recode+Death({rcd_l1}) == Recode({rc_l1}). Death penalty adds nothing.")
    elif rcd_l1 < rc_l1:
        print(f"  DEGRADED: Recode+Death({rcd_l1}) < Recode({rc_l1}). Penalty hurts exploration.")
    else:
        print(f"  MIXED: Recode+Death={rcd_l1} Recode={rc_l1} SP={sp_l1} AM={am_l1}")

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
