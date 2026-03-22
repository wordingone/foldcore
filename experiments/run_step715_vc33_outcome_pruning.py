"""
Step 715 — Outcome-based action pruning (graph-derived R3) on VC33.

R3 hypothesis: the substrate discovers its effective action space by tracking
which actions expand the graph (lead to new cells) vs produce self-loops.
COSMETIC actions: after min_probes, new_cell_count == 0 -> deprioritized.
NON-COSMETIC (structural) actions: at least once led to a new cell.
Argmin runs over structural actions only post-warmup.
Revival: COSMETIC action that leads to new cell gets revived.

Key difference from observation-based pruning (step 714):
  - 714 uses delta(obs): ℓ₀ change in raw pixels
  - 715 uses delta(graph): action led to node not previously in graph
  VC33 fails ℓ₀ (uniform delta=3.0 for all actions). Does it succeed on ℓ_π?

Kill criterion: cosmetic=0 (all actions structural, graph too fine-grained) → kill.
Signal: 2-3 structural actions survive AND bootloader passes.

Game: VC33, vc33/9851e02b. 5 seeds, 120K steps.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 4096
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MIN_VISITS_ALIAS = 3
WARMUP_STEPS = 500
MIN_PROBES = 10
MAX_STEPS = 120_001
N_SEEDS = 5

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68

MAGIC_GRID_IDX_A = 3 * 8 + 7  # gy=3, gx=7 -> (60,28) -- near (62,26)
MAGIC_GRID_IDX_B = 4 * 8 + 7  # gy=4, gx=7 -> (60,36) -- near (62,34)
MAGIC_UNIV_IDX_A = 4 + MAGIC_GRID_IDX_A
MAGIC_UNIV_IDX_B = 4 + MAGIC_GRID_IDX_B


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.flatten()  # 4096D


class OutcomeAD_Raw:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live_nodes = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0
        # Episode-local novelty tracking (resets per episode — hash space shifts with _mu)
        self.all_seen = set()
        # Lifetime action classification (persists across episodes)
        self.structural_actions = set(range(N_UNIV))
        self.cosmetic_actions = set()
        self.new_cell_count = [0] * N_UNIV
        self.total_count = [0] * N_UNIV
        self._probe_ptr = 0
        self.steps = 0

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x_raw = enc_raw(frame)
        self._mu_n += 1
        self._mu = self._mu + (x_raw - self._mu) / self._mu_n
        x = x_raw - self._mu
        n = self._node(x); fn = self._hash_fine(x)
        self.all_seen.add(n)
        self.live_nodes.add(n); self.t += 1
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

    def peek_node(self, frame):
        """Compute node hash without updating state. For outcome check after step."""
        x_raw = enc_raw(frame)
        x = x_raw - self._mu  # use current _mu (not yet updated for this frame)
        n = self._node(x)
        return n, n not in self.all_seen

    def act(self):
        self.steps += 1
        active = sorted(self.structural_actions)
        if not active:
            active = sorted(range(N_UNIV))  # fallback: use all
        if self.steps <= WARMUP_STEPS:
            idx = active[self._probe_ptr % len(active)]
            self._probe_ptr += 1
        else:
            if self._cn in self.aliased and self._fn is not None:
                idx = min(active, key=lambda a: sum(self.G_fine.get((self._fn, a), {}).values()))
            else:
                idx = min(active, key=lambda a: sum(self.G.get((self._cn, a), {}).values()))
        self._pn = self._cn; self._pfn = self._fn; self._pa = idx
        return idx

    def report_outcome(self, action_idx, next_node_is_new):
        """Record whether this action led to a new (never-seen) graph node."""
        self.total_count[action_idx] += 1
        if next_node_is_new:
            self.new_cell_count[action_idx] += 1
            # Revival: COSMETIC action that leads to new cell -> revive
            if action_idx in self.cosmetic_actions:
                self.cosmetic_actions.discard(action_idx)
                self.structural_actions.add(action_idx)
        # Mark cosmetic if probed enough with no new cells
        if (self.total_count[action_idx] >= MIN_PROBES
                and self.new_cell_count[action_idx] == 0
                and action_idx not in self.cosmetic_actions):
            self.cosmetic_actions.add(action_idx)
            self.structural_actions.discard(action_idx)

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0
        # all_seen resets: hash space shifts when _mu resets, old nodes non-comparable
        self.all_seen = set()
        # structural/cosmetic classification PERSISTS — substrate keeps what it learned

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live_nodes or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live_nodes.discard(n); did += 1
            if did >= 3: break


def run(seed, make):
    env = make()
    sub = OutcomeAD_Raw(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = UNIVERSAL_ACTIONS[action_idx]
        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}

        # Outcome check: did this action lead to a new graph node?
        if obs_new is not None:
            _, next_is_new = sub.peek_node(obs_new)
        else:
            next_is_new = False
        sub.report_outcome(action_idx, next_is_new)

        obs = obs_new
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    elapsed = time.time() - t_start
    structural = len(sub.structural_actions)
    cosmetic = len(sub.cosmetic_actions)
    struct_dir = len([a for a in sub.structural_actions if a < 4])
    struct_click = len([a for a in sub.structural_actions if a >= 4])
    magic_a_struct = MAGIC_UNIV_IDX_A in sub.structural_actions
    magic_b_struct = MAGIC_UNIV_IDX_B in sub.structural_actions
    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} structural={structural}(dir={struct_dir},click={struct_click}) "
          f"cosmetic={cosmetic} magic=({'Y' if magic_a_struct else 'N'},{'Y' if magic_b_struct else 'N'}) "
          f"aliased={len(sub.aliased)} t={elapsed:.1f}s", flush=True)
    return dict(seed=seed, l1=l1, structural=structural, cosmetic=cosmetic,
                struct_dir=struct_dir, struct_click=struct_click,
                magic_a=magic_a_struct, magic_b=magic_b_struct,
                aliased=len(sub.aliased))


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("VC33")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 715: Outcome-based AD on VC33, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"R3: graph-derived pruning (structural=leads to new cell, cosmetic=self-loop only)")
    print(f"Magic click indices: UNIV[{MAGIC_UNIV_IDX_A}]->(action {UNIVERSAL_ACTIONS[MAGIC_UNIV_IDX_A]}), "
          f"UNIV[{MAGIC_UNIV_IDX_B}]->(action {UNIVERSAL_ACTIONS[MAGIC_UNIV_IDX_B]})")
    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    boot_n = sum(1 for r in results if r['l1'])
    elapsed = time.time() - t_start
    print(f"Bootloader: {boot_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    for r in results:
        status = "PASS" if r['l1'] else "FAIL"
        print(f"  s{r['seed']:2d}: {status} structural={r['structural']}(dir={r['struct_dir']},click={r['struct_click']}) "
              f"cosmetic={r['cosmetic']} magic=({'Y' if r['magic_a'] else 'N'},{'Y' if r['magic_b'] else 'N'})")

    all_cosmetic_zero = all(r['cosmetic'] == 0 for r in results)
    magic_detected = sum(1 for r in results if r['magic_a'] or r['magic_b'])

    print(f"\nR3 result:")
    if all_cosmetic_zero:
        print(f"KILL: cosmetic=0 in all seeds — graph too fine, every action maps to unique cell")
    elif boot_n >= 3 and magic_detected >= 3:
        print(f"SIGNAL: bootloader {boot_n}/5 + magic detected {magic_detected}/5 — proceed to full chain (716)")
    elif magic_detected >= 3:
        print(f"PARTIAL: magic detected {magic_detected}/5 but bootloader fails — structural pruning works, nav broken")
    else:
        print(f"FAIL: magic not detected ({magic_detected}/5) and bootloader fails — outcome pruning insufficient")


if __name__ == "__main__":
    main()
