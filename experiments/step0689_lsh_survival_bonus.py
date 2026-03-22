"""
Step 689 — Death-edge inversion: survival bonus for L2.

674 substrate (transition-triggered dual-hash, k=12+k=20).
ADD: track death_count per (cell, action).
     survival_rate(cell, a) = 1 - death_count / total_count

Action selection (at all cells):
  score(a) = count(a) - BONUS * survival_rate(cell, a)
  action = argmin(score)
  BONUS = 10 (modest — breaks ties toward non-lethal actions)

Death detection: done=True AND no level change in that step.

Seed 8, 300s cap.
Report every 30s: L1, L2, death_edges, aliased_cells.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 2_000_001
PER_SEED_TIME = 300
MIN_VISITS_ALIAS = 3
SEED = 8
REPORT_INTERVAL = 30
BONUS = 10


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class TransitionTriggeredSurvival:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self.G_fine = {}; self.aliased = set()
        self.death_edges = {}  # (node, action) -> death count
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame); n = self._node(x); fn = self._hash_fine(x)
        self.live.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                d2 = self.G_fine.setdefault((self._pfn, self._pa), {})
                d2[fn] = d2.get(fn, 0) + 1
        self._px = x; self._cn = n; self._fn = fn
        if self.t % REFINE_EVERY == 0: self._refine()
        return n

    def _survival_rate(self, node, a):
        total = sum(self.G.get((node, a), {}).values())
        if total == 0: return 1.0  # unknown → assume safe
        deaths = self.death_edges.get((node, a), 0)
        return 1.0 - deaths / total

    def act(self):
        if self._cn in self.aliased and self._fn is not None:
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                count = sum(self.G_fine.get((self._fn, a), {}).values())
                survival = self._survival_rate(self._cn, a)
                score = count - BONUS * survival
                if score < best_s: best_s = score; best_a = a
            self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            count = sum(self.G.get((self._cn, a), {}).values())
            survival = self._survival_rate(self._cn, a)
            score = count - BONUS * survival
            if score < best_s: best_s = score; best_a = a
        self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a

    def record_death(self):
        """Call when done=True and no level change (agent died)."""
        if self._pn is not None and self._pa is not None:
            k = (self._pn, self._pa)
            self.death_edges[k] = self.death_edges.get(k, 0) + 1

    def on_reset(self): self._pn = None; self._pfn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live.discard(n); did += 1
            if did >= 3: break


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 689: Survival bonus (BONUS={BONUS}) on seed {SEED}, {PER_SEED_TIME}s cap")
    sub = TransitionTriggeredSurvival(seed=SEED * 1000)
    obs = env.reset(seed=SEED)
    level = 0; l1 = l2 = None; l1_aliased = None
    total_deaths = 0
    t_start = time.time(); next_report = REPORT_INTERVAL

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=SEED); sub.on_reset(); continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0

        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step; l1_aliased = len(sub.aliased)
                print(f"  L1 at step {l1}, aliased={l1_aliased}, t={time.time()-t_start:.1f}s", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  L2 at step {l2}, aliased={len(sub.aliased)}, deaths={total_deaths}, t={time.time()-t_start:.1f}s", flush=True)
            level = cl; sub.on_reset()

        if done:
            if cl == level:
                # Death: no level change
                sub.record_death()
                total_deaths += 1
            obs = env.reset(seed=SEED); sub.on_reset()

        elapsed = time.time() - t_start
        if elapsed >= next_report:
            death_edge_count = len(sub.death_edges)
            print(f"  t={elapsed:.0f}s step={step} aliased={len(sub.aliased)} "
                  f"deaths={total_deaths} death_edges={death_edge_count} L1={l1} L2={l2}", flush=True)
            next_report += REPORT_INTERVAL
        if elapsed > PER_SEED_TIME:
            break

    print(f"\n{'='*60}")
    print(f"L1={l1}  L2={l2}  aliased_final={len(sub.aliased)}  aliased_at_l1={l1_aliased}")
    print(f"total_deaths={total_deaths}  death_edges={len(sub.death_edges)}")
    if l2: print("BREAKTHROUGH: L2 reached!")
    elif l1: print("FINDING: L1 reached, L2 not — survival bonus didn't unlock L2")
    else: print("KILL: L1 not reached")


if __name__ == "__main__":
    main()
