"""
Step 671 — Splatter substrate (one frame -> predict the world).

Inspired by single-image Gaussian splatting: predict next frame for each action.
Choose action whose predicted next frame is MOST DIFFERENT from current frame.

1-step world model: predicted_next[cell][action] = running mean of frames
observed at successor cell after taking that action.

Action selection: argmax_a L2(current_frame, predicted_next[cell][a])
for explored actions (count >= 3). Fallback to argmin for unexplored.

NOT curiosity (prediction ERROR). This is TRANSITION MAGNITUDE:
"which action takes me somewhere that looks most different from here?"

First time the transition structure is used for action SELECTION, not counting.

10 seeds, 25s cap.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
MIN_PRED_COUNT = 3  # min transitions before using predicted frame

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SplatterRecode:
    """Graph + 1-step world model for action selection by transition magnitude."""

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}   # {(n,a): {successor: count}}
        self.C = {}   # {(n,a,n2): (sum_x, count)}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = dim
        self._cn = None
        # World model: cell -> action -> (mean_frame_sum, count)
        self.succ_frames = {}  # (cell, action) -> (sum_frame, count)
        self.deaths = 0
        self.steps_since_death = 0

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            # Update world model: successor of (prev_cell, prev_action) = current frame
            key = (self._pn, self._pa)
            if key not in self.succ_frames:
                self.succ_frames[key] = (np.zeros(self.dim, np.float64), 0)
            sf, sc = self.succ_frames[key]
            self.succ_frames[key] = (sf + x.astype(np.float64), sc + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        x = self._px  # current frame
        best_a = 0
        best_score = -1.0

        # First: check if any action has a prediction
        has_pred = False
        for a in range(N_A):
            key = (self._cn, a)
            if key in self.succ_frames and self.succ_frames[key][1] >= MIN_PRED_COUNT:
                has_pred = True
                break

        if has_pred:
            # argmax transition magnitude
            for a in range(N_A):
                key = (self._cn, a)
                if key in self.succ_frames and self.succ_frames[key][1] >= MIN_PRED_COUNT:
                    sf, sc = self.succ_frames[key]
                    predicted = (sf / sc).astype(np.float32)
                    novelty = float(np.linalg.norm(x - predicted))
                    if novelty > best_score:
                        best_score = novelty
                        best_a = a
                # fallback: if action unexplored, treat as infinite novelty
                elif self.succ_frames.get(key, (None, 0))[1] < MIN_PRED_COUNT:
                    # Unexplored action: very high novelty
                    edge_count = sum(self.G.get((self._cn, a), {}).values())
                    if edge_count == 0 and best_score < float('inf'):
                        best_score = float('inf')
                        best_a = a
        else:
            # No predictions yet: fall back to argmin
            best_s = float('inf')
            for a in range(N_A):
                s = sum(self.G.get((self._cn, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a

        self._pn = self._cn
        self._pa = best_a
        return best_a

    def on_reset(self, is_death=False):
        if is_death:
            self.deaths += 1
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


def run(seed, make):
    env = make()
    sub = SplatterRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue
        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            sub.on_reset()
        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset(is_death=True)
        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"

    death_rate = sub.deaths / max(go, 1)
    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} go={go} deaths={sub.deaths} "
          f"death_rate={death_rate:.2f}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, deaths=sub.deaths,
                death_rate=death_rate)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Splatter (transition magnitude): {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    avg_death = np.mean([r['death_rate'] for r in results])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                if r['l1'] and BASELINE_L1.get(r['seed'])]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0

    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}  avg_death_rate={avg_death:.3f}")
    print(f"avg_speedup={avg_ratio:.2f}x (vs argmin baseline)")

    if avg_death > 0.05:
        print(f"\nWARNING: death rate {avg_death:.1%} > 5% — noisy TV (death = max frame diff)")

    if l2_n > 0:
        print("BREAKTHROUGH: L2 reached — world model enables L2 navigation")
    elif l1_n >= 6 and avg_ratio > 1.5:
        print("FINDING: Splatter significantly faster than argmin — world model helps")
    elif l1_n >= 4:
        print(f"FINDING: Splatter reaches L1 ({l1_n}/10) — marginal signal")
    else:
        print("KILL: Transition magnitude fails — noisy TV or wrong direction")


if __name__ == "__main__":
    main()
