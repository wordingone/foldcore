"""
Step 577d -- Self-discovering substrate: windowed buffer evaluation.

Fixes 577c's argmin-masking problem (Leo mail 2171, 2026-03-20).

Key changes from 577c:
  1. 100% buffer navigation (not 80/20). Each window: buffer IS the navigator.
  2. 5000-step warmup: pure argmin builds the graph AND all buffers accumulate data.
  3. Rotate through buffers in 500-step windows. Count L1s per buffer.

Protocol:
  - Steps 0-4999: pure argmin (warmup)
  - Steps 5000+: rotate buf[0..4] in 500-step windows
    [buf0: 0-499] [buf1: 500-999] [buf2: 1000-1499] [buf3: 1500-1999] [buf4: 2000-2499] repeat
  - During each window: buffer provides navigation when it has a target; argmin otherwise
  - L1 during window[i] → l1_counts[i]++

After 195K post-warmup steps: 195K / 2500 = 78 rotations → each buffer gets 78 windows.
Expected signal: mode gets 10+ L1s, mean/var/diff/min get 0-5.

Kill: all l1_counts flat after 200K steps.
Signal: mode_l1 >> mean_l1 → mode map discovery confirmed.

5 seeds, TIME_CAP=280s each.
"""
import time
import numpy as np
import sys
from scipy.ndimage import label as ndlabel

K = 12
DIM = 256
N_A = 4
N_BUF = 5
BUF_NAMES = ['mode', 'mean', 'var', 'diff', 'min']
WARMUP_STEPS = 5000
WINDOW_STEPS = 500
MAX_STEPS = 200_000
TIME_CAP = 280
MIN_CC = 2
MAX_CC = 150


# ── target detection ──────────────────────────────────────────────────────────

def find_nearest_target(map2d, agent_yx):
    """
    Find nearest CC in a 2D map (threshold: mean + 0.5*std).
    Returns (cy, cx) or None.
    """
    if map2d is None or agent_yx is None:
        return None
    std = float(map2d.std())
    if std < 1e-6:
        return None
    thresh = float(map2d.mean()) + 0.5 * std
    binary = map2d >= thresh
    labeled, n = ndlabel(binary)
    if n == 0:
        return None
    ay, ax = agent_yx
    best_dist = 1e9
    bcy = bcx = None
    for cid in range(1, n + 1):
        region = labeled == cid
        sz = int(region.sum())
        if sz < MIN_CC or sz > MAX_CC:
            continue
        ys, xs = np.where(region)
        cy, cx = float(ys.mean()), float(xs.mean())
        d = (cy - ay)**2 + (cx - ax)**2
        if d < best_dist:
            best_dist = d; bcy = cy; bcx = cx
    return (bcy, bcx) if bcy is not None else None


# ── substrate ─────────────────────────────────────────────────────────────────

class WindowedSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        # Statistics buffers
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mean_a = np.zeros((64, 64), dtype=np.float32)
        self.m2 = np.zeros((64, 64), dtype=np.float32)
        self.diff_a = np.zeros((64, 64), dtype=np.float32)
        self.min_a = np.full((64, 64), 15, dtype=np.int32)

        self.n_frames = 0
        self.prev_arr = None
        self.agent_yx = None

        # Window tracking
        self.l1_counts = np.zeros(N_BUF, dtype=np.int32)
        self.win_counts = np.zeros(N_BUF, dtype=np.int32)  # windows per buffer
        self.fallback_counts = np.zeros(N_BUF, dtype=np.int32)  # fallback-to-argmin

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)

        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr)
            nc = int((diff > 0).sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff > 0)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
            self.diff_a += diff.astype(np.float32)

        self.prev_arr = arr.copy()
        self.n_frames += 1
        n = self.n_frames

        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1

        a = arr.astype(np.float32)
        delta = a - self.mean_a
        self.mean_a += delta / n
        delta2 = a - self.mean_a
        self.m2 += delta * delta2

        self.min_a = np.minimum(self.min_a, arr)

        x = arr.astype(np.float32).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
        x -= x.mean()
        cell = int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)
        self.cells.add(cell)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
        self._cn = cell

    def _get_map(self, buf_idx):
        if self.n_frames < 20:
            return None
        if buf_idx == 0:
            return np.argmax(self.freq, axis=2).astype(np.float32)
        elif buf_idx == 1:
            return self.mean_a.copy()
        elif buf_idx == 2:
            if self.n_frames < 2: return None
            return (self.m2 / max(self.n_frames - 1, 1)).astype(np.float32)
        elif buf_idx == 3:
            return self.diff_a.copy()
        elif buf_idx == 4:
            return self.min_a.astype(np.float32)
        return None

    def act_argmin(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action
        return action

    def act_buffer(self, buf_idx):
        """Navigate using buffer buf_idx. Fall back to argmin if no target."""
        if self.agent_yx is not None:
            target = find_nearest_target(self._get_map(buf_idx), self.agent_yx)
            if target is not None:
                ty, tx = target; ay, ax = self.agent_yx
                dy, dx = ty - ay, tx - ax
                action = (0 if dy < 0 else 1) if abs(dy) >= abs(dx) else (2 if dx < 0 else 3)
                self._pn = self._cn; self._pa = action
                return action
        # Fallback
        self.fallback_counts[buf_idx] += 1
        return self.act_argmin()

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = WindowedSub(lsh_seed=0)
    frame = [np.random.RandomState(1).randint(0, 16, (64, 64), dtype=np.uint8)]
    for _ in range(30):
        sub.observe(frame)
        sub.act_argmin()
    a = sub.act_buffer(0)
    assert 0 <= a < 4
    print(f"T0 PASS: action={a} fallback={sub.fallback_counts}")

    # Verify target detection
    m = np.zeros((64, 64), dtype=np.float32); m[20:24, 30:34] = 1.0
    t = find_nearest_target(m, (10.0, 10.0))
    assert t is not None
    print(f"T0 PASS: target={t}")


# ── seed runner ────────────────────────────────────────────────────────────────

def run_seed(mk, seed, time_cap=TIME_CAP):
    env = mk()
    sub = WindowedSub(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    prev_cl = 0; fresh = True
    l1_total = l2_total = go = step = 0
    t_start = time.time()

    while step < MAX_STEPS and time.time() - t_start < time_cap:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        sub.observe(obs)

        if step < WARMUP_STEPS:
            action = sub.act_argmin()
        else:
            post = step - WARMUP_STEPS
            buf_idx = (post // WINDOW_STEPS) % N_BUF
            win_start = post % WINDOW_STEPS == 0
            if win_start:
                sub.win_counts[buf_idx] += 1
            action = sub.act_buffer(buf_idx)

        obs, _, done, info = env.step(action)
        step += 1

        if done:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1_total += 1
            if step >= WARMUP_STEPS:
                post = step - WARMUP_STEPS
                buf_idx = (post // WINDOW_STEPS) % N_BUF
                sub.l1_counts[buf_idx] += 1
                if l1_total <= 5:
                    print(f"  s{seed} L1@{step} buf={BUF_NAMES[buf_idx]} "
                          f"l1_counts={sub.l1_counts}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2_total += 1
        prev_cl = cl

    elapsed = time.time() - t_start
    fallback_rate = sub.fallback_counts / np.maximum(sub.win_counts * WINDOW_STEPS, 1)
    print(f"  s{seed}: L1={l1_total} L2={l2_total} go={go} step={step} "
          f"cells={len(sub.cells)} {elapsed:.0f}s", flush=True)
    print(f"    l1_counts: {dict(zip(BUF_NAMES, sub.l1_counts))}", flush=True)
    print(f"    win_counts: {dict(zip(BUF_NAMES, sub.win_counts))}", flush=True)
    print(f"    fallback%:  {dict(zip(BUF_NAMES, (fallback_rate*100).round(1)))}", flush=True)
    return dict(seed=seed, l1=l1_total, l2=l2_total, go=go, steps=step,
                cells=len(sub.cells), l1_counts=sub.l1_counts.copy(),
                win_counts=sub.win_counts.copy())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"\nStep 577d: Windowed buffer evaluation", flush=True)
    print(f"  WARMUP={WARMUP_STEPS} WINDOW={WINDOW_STEPS} N_BUF={N_BUF}", flush=True)

    results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP HIT"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=TIME_CAP)
        results.append(r)

    # Aggregate
    total_l1 = np.sum([r['l1_counts'] for r in results], axis=0)
    total_wins = np.sum([r['win_counts'] for r in results], axis=0)
    any_l1 = sum(1 for r in results if r['l1'] > 0)

    print(f"\n{'='*60}")
    print("STEP 577d: Windowed buffer evaluation")
    for r in results:
        print(f"  s{r['seed']}: L1={r['l1']} "
              f"l1_counts={dict(zip(BUF_NAMES, r['l1_counts']))}")
    print(f"\nTotal L1s per buffer: {dict(zip(BUF_NAMES, total_l1))}")
    print(f"Total windows per buffer: {dict(zip(BUF_NAMES, total_wins))}")
    print(f"L1 seeds: {any_l1}/{len(results)}")

    if total_wins.max() > 0:
        l1_rate = total_l1 / np.maximum(total_wins, 1)
        print(f"L1 rate per window: {dict(zip(BUF_NAMES, l1_rate.round(3)))}")
        winner = BUF_NAMES[int(np.argmax(l1_rate))]
        winner_rate = float(l1_rate.max())
        if winner_rate > 0.05:   # >5% of windows produced L1
            print(f"\nDISCOVERED: buf[{winner}] highest L1 rate ({winner_rate:.3f} L1/window)")
        else:
            print(f"\nFAIL: No buffer achieves >5% L1 rate per window. "
                  f"Best: {winner}={winner_rate:.3f}")


if __name__ == "__main__":
    main()
