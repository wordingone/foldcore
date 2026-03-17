#!/usr/bin/env python3
"""
Step 414 -- Sequential resolution dedication.

Try each resolution in descending order (64→32→16→8), 50K steps each.
Fresh codebook per segment. EXACT Step 353 baseline (substrate + centered_enc).
If a resolution finds a level → lock it. Done.

Total: up to 200K steps.
LS20.
Script: scripts/run_step414_sequential_res.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS_PER_RES = 50000

RESOLUTIONS = [16, 8]  # skip 64/32 (proven dead), test 16 then 8


class CompressedFold:
    """EXACT Step 353 baseline. V stores normalized. Thresh updates every step."""
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def process(self, x, nc, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label

        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(n_cls, nc), device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]; scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        prediction = scores[:nc].argmin().item()
        spawn_label = label if label is not None else prediction
        target_mask = (self.labels == prediction)

        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([spawn_label], device=self.device)])
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)

        self._update_thresh()
        return prediction


def avgpool(frame, N):
    """Pool 64x64 frame to NxN."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    if N == 64:
        return arr.flatten()
    k = 64 // N
    return arr.reshape(N, k, N, k).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    """Step 353 centered encoding: normalize then subtract codebook mean."""
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def run_segment(arc, game_id, res, max_steps, verbose=True):
    """Run one resolution segment. Returns (levels_found, stats_dict)."""
    from arcengine import GameState

    d = res * res
    fold = CompressedFold(d=d, k=3)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)

    ts = 0; go = 0; lvls = 0; seeded = False
    action_counts = {}

    while ts < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            if verbose:
                print(f"      WIN at step {ts}!", flush=True)
            break

        pooled = avgpool(obs.frame, res)
        x = centered_enc(pooled, fold)

        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]
            fold._force_add(x, i)
            obs = env.step(env.action_space[i]); ts += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= na: seeded = True
            continue
        if not seeded: seeded = True

        c = fold.process(x, na)
        action = env.action_space[c % na]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        ol = obs.levels_completed
        obs = env.step(action); ts += 1
        if obs is None: break

        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            if verbose:
                print(f"      LEVEL {lvls} at step {ts}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}  go={go}", flush=True)

        if ts % 10000 == 0 and verbose:
            dom = max(action_counts.values()) / sum(action_counts.values()) * 100 if action_counts else 0
            print(f"      [step {ts:5d}] cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}"
                  f"  dom={dom:.0f}%  levels={lvls}  go={go}", flush=True)

    dom = max(action_counts.values()) / sum(action_counts.values()) * 100 if action_counts else 0
    stats = {
        'steps': ts, 'game_overs': go, 'levels': lvls,
        'cb_size': fold.V.shape[0], 'thresh': fold.thresh,
        'dom': dom, 'actions': action_counts,
    }
    return lvls, stats


def main():
    t0 = time.time()
    print(f"Step 414 -- Sequential resolution dedication. Exact Step 353 (substrate + centered_enc).", flush=True)
    print(f"Device: {DEVICE}  resolutions={RESOLUTIONS}  steps_per_res={STEPS_PER_RES}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    results = {}
    locked_res = None
    total_steps = 0

    for res in RESOLUTIONS:
        print(f"  === {res}x{res} segment (50K steps) ===", flush=True)
        seg_t0 = time.time()
        lvls, stats = run_segment(arc, ls20.game_id, res, STEPS_PER_RES)
        seg_elapsed = time.time() - seg_t0
        total_steps += stats['steps']
        results[res] = stats

        print(f"      SEGMENT DONE: levels={lvls}  cb={stats['cb_size']}"
              f"  thresh={stats['thresh']:.3f}  dom={stats['dom']:.0f}%"
              f"  go={stats['game_overs']}  elapsed={seg_elapsed:.1f}s", flush=True)
        print(flush=True)

        if lvls > 0:
            locked_res = res
            print(f"  >>> LOCKED: {res}x{res} found {lvls} level(s)! <<<", flush=True)
            print(flush=True)
            break

    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 414 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total steps: {total_steps}  Locked: {locked_res}", flush=True)
    print(flush=True)

    for res in RESOLUTIONS:
        if res not in results: continue
        s = results[res]
        mark = " <-- LOCKED" if res == locked_res else ""
        print(f"  {res:2d}x{res:2d}: levels={s['levels']}  steps={s['steps']}"
              f"  cb={s['cb_size']}  thresh={s['thresh']:.3f}"
              f"  dom={s['dom']:.0f}%  go={s['game_overs']}{mark}", flush=True)

    if locked_res:
        print(f"\nPASS: Resolution {locked_res}x{locked_res} discovered via sequential trial!", flush=True)
    else:
        print(f"\nKILL: No resolution found a level in {total_steps} total steps.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__': main()
