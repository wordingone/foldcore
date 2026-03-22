"""
Step 657 — Action sequence entropy before L1.

Is L1 reached by structured action sequences or random coverage?

At L1 trigger, extract last 50 actions. Compute:
  1. Action entropy H (max=2.0 = uniform, min=0 = one action)
  2. Longest consecutive same-action run
  3. First-order transition entropy H(a_t | a_{t-1})

High entropy (~2.0): L1 stumbled into via uniform exploration.
Low entropy (<1.5): L1 requires structured sequences.
"""
import numpy as np
import time
import sys
from collections import deque, Counter

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
WINDOW = 50


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim

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
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

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
            self.ns += 1
            did += 1
            if did >= 3:
                break


def action_entropy(actions):
    """Entropy of action distribution in window."""
    if not actions:
        return 0.0
    cnt = Counter(actions)
    n = len(actions)
    p = np.array([cnt.get(a, 0) / n for a in range(N_A)], dtype=np.float64)
    return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))


def transition_entropy(actions):
    """First-order transition entropy H(a_t | a_{t-1})."""
    if len(actions) < 2:
        return 0.0
    trans = Counter(zip(actions[:-1], actions[1:]))
    prev_cnt = Counter(actions[:-1])
    h = 0.0
    for (prev, cur), cnt in trans.items():
        p_joint = cnt / (len(actions) - 1)
        p_prev = prev_cnt[prev] / (len(actions) - 1)
        h -= p_joint * np.log2(p_joint / (p_prev + 1e-15) + 1e-15)
    return max(0.0, h)


def longest_run(actions):
    """Longest consecutive same-action run."""
    if not actions:
        return 0
    max_run = 1
    cur_run = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()
    action_history = deque(maxlen=WINDOW)
    l1_window = None

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        action_history.append(action)
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                l1_window = list(action_history)
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    if l1 is not None and l1_window:
        h = action_entropy(l1_window)
        th = transition_entropy(l1_window)
        lr = longest_run(l1_window)
        cnt = Counter(l1_window)
        print(f"  s{seed}: L1={l1} H={h:.3f} trans_H={th:.3f} max_run={lr} "
              f"dist={[cnt.get(a,0) for a in range(N_A)]}", flush=True)
    else:
        print(f"  s{seed}: L1=None", flush=True)

    return dict(
        seed=seed, l1=l1,
        window=l1_window,
        H=action_entropy(l1_window) if l1_window else None,
        trans_H=transition_entropy(l1_window) if l1_window else None,
        max_run=longest_run(l1_window) if l1_window else None,
    )


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Action entropy before L1: {N_SEEDS} seeds, {PER_SEED_TIME}s cap, window={WINDOW}")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_r = [r for r in results if r['l1'] is not None]
    print(f"L1 reached: {len(l1_r)}/{N_SEEDS}")

    if l1_r:
        h_vals = [r['H'] for r in l1_r if r['H'] is not None]
        th_vals = [r['trans_H'] for r in l1_r if r['trans_H'] is not None]
        mr_vals = [r['max_run'] for r in l1_r if r['max_run'] is not None]

        print(f"Action entropy H: avg={np.mean(h_vals):.3f} "
              f"(max=2.0=uniform, min=0=one action)")
        print(f"Transition entropy: avg={np.mean(th_vals):.3f}")
        print(f"Max consecutive run: avg={np.mean(mr_vals):.1f}")

        if np.mean(h_vals) > 1.8:
            verdict = "RANDOM COVERAGE: L1 stumbled into via near-uniform action selection"
        elif np.mean(h_vals) < 1.5:
            verdict = "STRUCTURED: L1 requires specific action sequences (low entropy)"
        else:
            verdict = f"MIXED: H={np.mean(h_vals):.3f}, partial structure"
        print(f"\n{verdict}")


if __name__ == "__main__":
    main()
