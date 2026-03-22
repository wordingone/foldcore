"""
Step 571 — Candidate sweep for L2. R1-compliant.

Phase 1 (L1): Mode map + rare-color targeting (same as Step 567). Reach L1.
              Record which target triggered L1 = exit_target.
Phase 2 (L2): Cycle non-exit candidates. Each episode: navigate candidate[i % N],
              then navigate exit_target. Track which candidate triggers L2.

Budget: 5 seeds, 200K steps each (5-min cap per seed).
Kill: L1=0/5 OR L2=0/5 after exhausting candidates.
Find: L2>0 -> first ever L2.
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel

N_A = 4
K = 16
FG_DIM = 4096
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05
MODE_EVERY = 200
WARMUP = 100
RARE_THRESH = 0.05
MIN_CLUSTER = 2
MAX_CLUSTER = 20
VISIT_DIST = 4
REDETECT_EVERY = 500


def dir_action(ty, tx, ay, ax):
    """0=UP, 1=DOWN, 2=LEFT, 3=RIGHT."""
    dy = ty - ay
    dx = tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    else:
        return 2 if dx < 0 else 3


def find_rare_clusters(mode_arr):
    total = mode_arr.size
    rare_thresh_px = total * RARE_THRESH
    colors, counts = np.unique(mode_arr, return_counts=True)
    rare_colors = colors[counts < rare_thresh_px]
    clusters = []
    for color in rare_colors:
        mask = (mode_arr == color)
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if sz < MIN_CLUSTER or sz > MAX_CLUSTER:
                continue
            ys, xs = np.where(region)
            clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                             'color': int(color), 'size': sz})
    return clusters


class Sub571:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, FG_DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self._last_visit = {}
        # Background model
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        # Targeting (L1 phase)
        self.targets = []
        self.visited = []
        self.agent_yx = None
        self.prev_arr = None
        self._curr_arr = None
        self._steps_since_detect = REDETECT_EVERY
        # Phase tracking
        self.phase = "l1"           # "l1" or "l2"
        self.exit_target = None
        self.candidates = []        # non-exit targets for L2
        self.episode_in_l2 = 0
        self.ep_seq = []            # [candidate, exit] for current episode
        self.ep_seq_idx = 0
        self.current_candidate = None
        # Stats
        self.target_actions = 0
        self.fb_actions = 0
        self.n_targets_found = 0
        self.candidates_tested = 0

    def _update_bg(self, arr):
        r = np.arange(64)[:, None]
        c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1
        self.n_frames += 1
        if self.n_frames % MODE_EVERY == 0:
            self.mode = np.argmax(self.freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        return (arr != self.mode).astype(np.float32).flatten()

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
        arr = np.array(frame[0], dtype=np.int32)
        self._update_bg(arr)
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
        self.prev_arr = arr.copy()
        self._curr_arr = arr
        if self.n_frames < WARMUP:
            x = arr.astype(np.float32).flatten() / 15.0
            x = x - x.mean()
        else:
            x = self._fg_enc(arr)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(FG_DIM, np.float64), 0))
            self.C[k_key] = (s + x.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        self._steps_since_detect += 1
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def on_l1(self):
        """Called when L1 is reached. Identify exit_target, set up L2 candidates."""
        # Detect/refresh targets now
        if self.n_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
        # Exit target = nearest target to agent position at L1 moment
        if self.agent_yx and self.targets:
            ay, ax = self.agent_yx
            self.exit_target = min(
                self.targets,
                key=lambda t: (t['cy'] - ay) ** 2 + (t['cx'] - ax) ** 2
            )
            self.candidates = [
                t for t in self.targets
                if not (abs(t['cy'] - self.exit_target['cy']) < 1 and
                        abs(t['cx'] - self.exit_target['cx']) < 1)
            ]
        self.phase = "l2"
        self.episode_in_l2 = 0
        self.candidates_tested = 0
        self._build_ep_seq()

    def _build_ep_seq(self):
        """Set up target sequence for current L2 episode: [candidate_i, exit]."""
        if not self.candidates or self.exit_target is None:
            self.ep_seq = []
            self.current_candidate = None
            return
        ci = self.episode_in_l2 % len(self.candidates)
        self.current_candidate = self.candidates[ci]
        self.ep_seq = [self.current_candidate, self.exit_target]
        self.ep_seq_idx = 0

    def _argmin_action(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        self.fb_actions += 1
        return action

    def act(self):
        if self.phase == "l1":
            return self._act_l1()
        else:
            return self._act_l2()

    def _act_l1(self):
        # Redetect targets periodically
        if self._steps_since_detect >= REDETECT_EVERY and self.n_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
            self._steps_since_detect = 0

        if self.targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None
            best_dist = 1e9
            for t in self.targets:
                visited = any(
                    ((t['cy'] - vy) ** 2 + (t['cx'] - vx) ** 2) < VISIT_DIST ** 2
                    for vy, vx in self.visited
                )
                if visited:
                    continue
                dist = ((t['cy'] - ay) ** 2 + (t['cx'] - ax) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = t

            if best is not None:
                if best_dist < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    action = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn
                    self._pa = action
                    self.target_actions += 1
                    return action

        return self._argmin_action()

    def _act_l2(self):
        if not self.ep_seq or self.agent_yx is None:
            return self._argmin_action()

        target = self.ep_seq[self.ep_seq_idx]
        ay, ax = self.agent_yx
        dist = ((target['cy'] - ay) ** 2 + (target['cx'] - ax) ** 2) ** 0.5

        if dist < VISIT_DIST:
            self.ep_seq_idx += 1
            if self.ep_seq_idx >= len(self.ep_seq):
                # Sequence exhausted, fall back to argmin
                return self._argmin_action()
            target = self.ep_seq[self.ep_seq_idx]

        action = dir_action(target['cy'], target['cx'], ay, ax)
        self._pn = self._cn
        self._pa = action
        self.target_actions += 1
        return action

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None
        self.visited = []
        self._steps_since_detect = REDETECT_EVERY
        if self.phase == "l2":
            self.episode_in_l2 += 1
            self.candidates_tested = self.episode_in_l2
            self._build_ep_seq()

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)

    def stats(self):
        return len(self.live), len(self.ref), len(self.G)


def t0():
    rng = np.random.RandomState(0)
    # Test rare clusters
    mode = np.zeros((64, 64), dtype=np.int32)
    mode[10:12, 10:15] = 7
    clusters = find_rare_clusters(mode)
    colors = {c['color'] for c in clusters}
    assert 7 in colors, f"Should find rare color 7: {colors}"
    assert 0 not in colors, "Background should be filtered"
    # dir_action
    assert dir_action(2, 5, 8, 5) == 0
    assert dir_action(10, 5, 4, 5) == 1
    assert dir_action(5, 2, 5, 8) == 2
    assert dir_action(5, 8, 5, 2) == 3
    # Sub571 smoke
    sub = Sub571(seed=0)
    for _ in range(5):
        f = [rng.randint(0, 16, (64, 64))]
        sub.observe(f)
        sub.act()
    sub.on_reset()
    assert sub.visited == []
    assert sub.phase == "l1"
    print("T0 PASS")


def main():
    t0()
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 5
    per_seed_cap = 300  # 5-min cap per seed
    R = []
    t_start = time.time()

    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        env = mk()
        sub = Sub571(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        candidate_at_l2 = None
        go = 0
        seed_start = time.time()

        for step in range(1, 200_001):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_reset()
                continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)
            if done:
                go += 1
                obs = env.reset(seed=seed)
                sub.on_reset()
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                if cl == 1 and l1_step is None:
                    l1_step = step
                    sub.on_l1()
                    n_cands = len(sub.candidates)
                    exit_t = sub.exit_target
                    print(f"  s{seed} L1@{step} tgt={sub.n_targets_found} "
                          f"exit=({exit_t['cy']:.0f},{exit_t['cx']:.0f}) "
                          f"candidates={n_cands} go={go}", flush=True)
                sub.on_reset()
                if cl == 2 and l2_step is None:
                    l2_step = step
                    candidate_at_l2 = sub.current_candidate
                    print(f"  s{seed} L2@{step}! candidate={candidate_at_l2} "
                          f"episode={sub.episode_in_l2} "
                          f"candidates_tested={sub.candidates_tested}", flush=True)
            if step % 25_000 == 0:
                el = time.time() - seed_start
                print(f"  s{seed} @{step} phase={sub.phase} ep_l2={sub.episode_in_l2} "
                      f"cands={len(sub.candidates)} go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} seed cap hit @{step}", flush=True)
                break

        nc, ns, ne = sub.stats()
        R.append(dict(
            seed=seed, l1=l1_step, l2=l2_step,
            cells=nc, go=go, steps=step, level=level,
            n_targets=sub.n_targets_found,
            target_actions=sub.target_actions,
            fb=sub.fb_actions,
            candidates_tested=sub.candidates_tested,
            candidate_at_l2=candidate_at_l2,
        ))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    print(f"\nResults (Step 571 — candidate sweep for L2):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  tgt={r['n_targets']:>3}  "
              f"cands_tested={r['candidates_tested']:>3}  "
              f"ta={r['target_actions']:>6}")
        if r['l2']:
            print(f"         candidate_at_l2={r['candidate_at_l2']}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps (Step 567 baseline: 468)")
    avg_cands = np.mean([r['candidates_tested'] for r in R])
    print(f"Avg candidates tested: {avg_cands:.1f}")

    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! FIRST L2 EVER. Candidate sweep works!")
    elif l1n == 0:
        print(f"\nKILL: L1=0/{len(R)}. Regression from Step 567.")
    else:
        print(f"\nKILL: L1={l1n}/{len(R)}, L2=0. "
              f"Candidates tested but palette not found or routing failed.")


if __name__ == "__main__":
    main()
