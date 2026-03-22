"""
Step 571 — Candidate sweep: one rare target + exit per episode.

Based on Step 567 (mode map rare-color targeting, L1=5/5@468).
Step 568-569 showed visit-all-6 exhausts 129-step budget.

Fix: each episode, navigate to ONE non-exit target, then exit.
Cycle through candidates across episodes. If target[i] is the
energy palette, touching it refills energy -> can reach exit = L2.

Budget: 2 targets = ~60-80 steps, well within 129.
After first L1 (exit identified), cycle: [t0->exit], [t1->exit], ...

Kill: L1 < 3/5 (targeting hurts baseline)
Find: L2 > 0 (FIRST EVER L2)
5-min cap. LS20. 5 seeds.
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


class RecodeCandidate:
    """Mode map + candidate sweep: one non-exit target + exit per episode."""

    def __init__(self, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, FG_DIM).astype(np.float32)
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
        # Targeting
        self.targets = []
        self.exit_target = None       # identified after first L1
        self.candidate_idx = 0        # which non-exit target to try this episode
        self.non_exit_targets = []    # all targets except exit
        self.phase = 'explore'        # explore -> target_candidate -> go_exit
        self.agent_yx = None
        self.prev_arr = None
        self._curr_arr = None
        self._steps_since_detect = REDETECT_EVERY
        self.episode_step = 0
        # Stats
        self.target_actions = 0
        self.fb_actions = 0
        self.n_targets_found = 0
        self.l1_count = 0
        self.episodes = 0

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

    def _detect_targets(self):
        if self.n_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
            self._steps_since_detect = 0
            # Rebuild non-exit list if exit is known
            if self.exit_target is not None:
                ex = self.exit_target
                self.non_exit_targets = [
                    t for t in self.targets
                    if ((t['cy'] - ex['cy'])**2 + (t['cx'] - ex['cx'])**2) > VISIT_DIST**2
                ]

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
        self.episode_step += 1
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

    def act(self):
        # Redetect targets periodically
        if self._steps_since_detect >= REDETECT_EVERY and self.n_frames >= WARMUP:
            self._detect_targets()

        # Phase logic for candidate sweep
        if self.exit_target is not None and self.non_exit_targets and self.agent_yx is not None:
            ay, ax = self.agent_yx

            if self.phase == 'target_candidate':
                # Navigate to current candidate
                idx = self.candidate_idx % len(self.non_exit_targets)
                tgt = self.non_exit_targets[idx]
                dist = ((tgt['cy'] - ay)**2 + (tgt['cx'] - ax)**2) ** 0.5
                if dist < VISIT_DIST:
                    # Reached candidate, now go to exit
                    self.phase = 'go_exit'
                else:
                    action = dir_action(tgt['cy'], tgt['cx'], ay, ax)
                    self._pn = self._cn
                    self._pa = action
                    self.target_actions += 1
                    return action

            if self.phase == 'go_exit':
                # Navigate to exit
                ex = self.exit_target
                dist = ((ex['cy'] - ay)**2 + (ex['cx'] - ax)**2) ** 0.5
                if dist < VISIT_DIST:
                    # At exit, will reset soon
                    pass
                else:
                    action = dir_action(ex['cy'], ex['cx'], ay, ax)
                    self._pn = self._cn
                    self._pa = action
                    self.target_actions += 1
                    return action

        elif self.targets and self.agent_yx is not None and self.exit_target is None:
            # Pre-L1: greedy nearest (Step 567 approach)
            ay, ax = self.agent_yx
            best, best_dist = None, 1e9
            for t in self.targets:
                dist = ((t['cy'] - ay)**2 + (t['cx'] - ax)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = t
            if best is not None and best_dist >= VISIT_DIST:
                action = dir_action(best['cy'], best['cx'], ay, ax)
                self._pn = self._cn
                self._pa = action
                self.target_actions += 1
                return action

        # Fallback: argmin
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        self.fb_actions += 1
        return action

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None
        self._steps_since_detect = REDETECT_EVERY
        self.episodes += 1
        self.episode_step = 0

        # After first L1, identify exit as the target we were near at reset
        # (heuristic: whichever target was closest when level completed)
        # For candidate sweep: cycle to next candidate
        if self.exit_target is not None:
            self.candidate_idx += 1
            self.phase = 'target_candidate'
        else:
            # Check if we just got L1 (l1_count tracks externally)
            # For now, after enough episodes with targets, pick most-visited as exit
            self.phase = 'explore'

    def on_level(self, level):
        """Called externally when a level is reached."""
        if level >= 1 and self.exit_target is None and self.targets:
            # The target nearest to current agent pos is likely the exit
            if self.agent_yx is not None:
                ay, ax = self.agent_yx
                best, best_dist = None, 1e9
                for t in self.targets:
                    dist = ((t['cy'] - ay)**2 + (t['cx'] - ax)**2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best = t
                if best is not None:
                    self.exit_target = best
                    self.non_exit_targets = [
                        t for t in self.targets
                        if ((t['cy'] - best['cy'])**2 + (t['cx'] - best['cx'])**2) > VISIT_DIST**2
                    ]
                    self.phase = 'target_candidate'
                    self.candidate_idx = 0
            self.l1_count += 1

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
        return {
            'live': len(self.live),
            'refined': len(self.ref),
            'edges': len(self.G),
            'targets': self.n_targets_found,
            'exit_known': self.exit_target is not None,
            'candidate_idx': self.candidate_idx,
            'target_actions': self.target_actions,
            'fb_actions': self.fb_actions,
            'episodes': self.episodes,
        }
