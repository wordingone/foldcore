#!/usr/bin/env python3
"""
Step 415 -- Unified click-grid substrate. ALL games. Zero game knowledge.

The substrate ALWAYS uses an NxN click grid. Never reads the game's action space.
Game provides: frame (64x64). Substrate provides: click position (x, y).

- Visual: 16x16 avgpool + centered_enc → 256D
- Actions: NxN click grid = N² discrete regions. Click center of region.
- Substrate: Step 353 baseline (F.normalize, cosine, K=3, argmin)
- centered_enc: normalize then subtract codebook mean
- Codebook: UNCAPPED, thresh updates every step (full for ≤500, subsampled >500)
- Sequential grid trial: N ∈ {4, 8}. 50K per N. Lock on level.

Usage: python run_step415_unified.py [game_id]
  game_id: ls20, ft09, vc33, or "all" (default: all)
Script: scripts/run_step415_unified.py
"""

import sys, time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 50000; THRESH_SAMPLE = 500

GRID_SIZES = [4, 8]  # sequential trial: 4x4=16 actions, 8x8=64 actions


def action_to_click(action_idx, grid_N):
    """Convert grid action index to (x, y) click coordinates (center of region)."""
    region_y = action_idx // grid_N
    region_x = action_idx % grid_N
    cell = 64 // grid_N
    click_x = region_x * cell + cell // 2
    click_y = region_y * cell + cell // 2
    return click_x, click_y


class CompressedFold:
    """Step 353 baseline. Uncapped. Thresh updates every step."""
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
        if n <= THRESH_SAMPLE:
            G = self.V @ self.V.T
            G.fill_diagonal_(-float('inf'))
            self.thresh = float(G.max(dim=1).values.median())
        else:
            idx = torch.randperm(n, device=self.device)[:THRESH_SAMPLE]
            S = self.V[idx] @ self.V[idx].T
            S.fill_diagonal_(-float('inf'))
            self.thresh = float(S.max(dim=1).values.median())

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


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def run_game_grid(arc, game_id, grid_N, max_steps=MAX_STEPS, verbose=True):
    """Run one game with one grid size. Returns stats dict."""
    from arcengine import GameAction, GameState

    nc = grid_N * grid_N  # number of click regions
    fold = CompressedFold(d=256, k=3)
    env = arc.make(game_id); obs = env.reset()

    ts = 0; go = 0; lvls = 0; seeded = False
    action_counts = {}

    while ts < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            if verbose: print(f"      WIN at step {ts}!", flush=True)
            break

        pooled = avgpool16(obs.frame)
        x = centered_enc(pooled, fold)

        # Seed: one frame per action class (cycle through grid regions)
        if not seeded and fold.V.shape[0] < nc:
            i = fold.V.shape[0]
            fold._force_add(x, i)
            cx, cy = action_to_click(i, grid_N)
            obs = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy}); ts += 1
            action_counts[i] = action_counts.get(i, 0) + 1
            if fold.V.shape[0] >= nc: seeded = True
            continue
        if not seeded: seeded = True

        c = fold.process(x, nc)
        cx, cy = action_to_click(c, grid_N)
        action_counts[c] = action_counts.get(c, 0) + 1
        ol = obs.levels_completed
        obs = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy}); ts += 1
        if obs is None: break

        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            if verbose:
                print(f"      LEVEL {lvls} at step {ts}  grid={grid_N}x{grid_N}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}  go={go}", flush=True)

        if ts % 5000 == 0 and verbose:
            vals = action_counts.values()
            dom = max(vals) / sum(vals) * 100 if vals else 0
            unique_used = sum(1 for v in vals if v > 0)
            print(f"      [step {ts:5d}] cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}"
                  f"  dom={dom:.0f}%  used={unique_used}/{nc}  levels={lvls}  go={go}", flush=True)

    vals = action_counts.values()
    dom = max(vals) / sum(vals) * 100 if vals else 0
    return {
        'game': game_id, 'grid_N': grid_N, 'n_actions': nc,
        'steps': ts, 'game_overs': go, 'levels': lvls,
        'cb_size': fold.V.shape[0], 'thresh': fold.thresh,
        'dom': dom, 'action_counts': action_counts,
    }


def main():
    t0 = time.time()
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print(f"Step 415 -- Unified click-grid substrate. Zero game knowledge.", flush=True)
    print(f"Device: {DEVICE}  max_steps={MAX_STEPS}  grids={GRID_SIZES}  target={target}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade(); games = arc.get_environments()

    if target == 'all':
        game_ids = [g.game_id for g in games]
    else:
        game_ids = [g.game_id for g in games if target.lower() in g.game_id.lower()]
        if not game_ids:
            print(f"No game matching '{target}'", flush=True)
            return

    all_results = []
    for gid in game_ids:
        print(f"  === {gid} ===", flush=True)
        locked_grid = None
        for grid_N in GRID_SIZES:
            print(f"    --- grid {grid_N}x{grid_N} ({grid_N**2} actions) ---", flush=True)
            gt0 = time.time()
            stats = run_game_grid(arc, gid, grid_N)
            elapsed = time.time() - gt0
            all_results.append(stats)
            print(f"      DONE: levels={stats['levels']}  cb={stats['cb_size']}"
                  f"  thresh={stats['thresh']:.3f}  dom={stats['dom']:.0f}%"
                  f"  go={stats['game_overs']}  elapsed={elapsed:.1f}s", flush=True)
            print(flush=True)

            if stats['levels'] > 0:
                locked_grid = grid_N
                print(f"    >>> LOCKED: {grid_N}x{grid_N} grid found {stats['levels']} level(s)! <<<", flush=True)
                print(flush=True)
                break

        if not locked_grid:
            print(f"    No grid found a level for {gid}.", flush=True)
            print(flush=True)

    total_elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 415 SUMMARY", flush=True)
    print("=" * 60, flush=True)

    for s in all_results:
        mark = " PASS" if s['levels'] > 0 else " KILL"
        print(f"  {s['game']:20s} grid={s['grid_N']}x{s['grid_N']:d}"
              f"  levels={s['levels']}  steps={s['steps']}"
              f"  cb={s['cb_size']}  dom={s['dom']:.0f}%{mark}", flush=True)

    games_passed = len(set(s['game'] for s in all_results if s['levels'] > 0))
    print(f"\nGames with levels: {games_passed}/{len(game_ids)}", flush=True)
    print(f"Elapsed: {total_elapsed:.2f}s", flush=True)


if __name__ == '__main__': main()
