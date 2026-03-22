#!/usr/bin/env python3
"""
Step 370 -- Local timing game: can process() discover temporal patterns?

10x10 grid. Flash appears at random cell for 3/20 steps.
CLICK during flash = +1 (level). CLICK outside flash = game over.
WAIT always safe. 50 steps/life.

Kill: does substrate learn CLICK-when-flash, WAIT-otherwise?
Script: scripts/run_step370_local_timing.py
"""

import time
import random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 100  # 10x10

# Actions
WAIT = 0
CLICK = 1
N_ACTS = 2


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        sims = self.V[idx] @ self.V.T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def process_novelty(self, x, n_cls):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            return 0
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([pred], device=self.device)])
            self._update_thresh()
        else:
            ts = sims.clone(); ts[~tm] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return pred


class TimingGame:
    def __init__(self, grid_size=10, flash_period=20, flash_duration=3, life_steps=50):
        self.grid_size = grid_size
        self.flash_period = flash_period
        self.flash_duration = flash_duration
        self.life_steps = life_steps
        self.reset()

    def reset(self):
        self.step_in_life = 0
        self.flash_cell = random.randint(0, self.grid_size * self.grid_size - 1)
        self.phase = random.randint(0, self.flash_period - 1)  # random phase offset
        return self._get_frame()

    def _is_flash(self):
        return (self.step_in_life + self.phase) % self.flash_period < self.flash_duration

    def _get_frame(self):
        grid = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        if self._is_flash():
            grid[self.flash_cell] = 1.0
        return grid

    def step(self, action):
        """Returns (frame, reward, done, info)."""
        flash = self._is_flash()

        if action == CLICK:
            if flash:
                # Correct click during flash
                self.step_in_life += 1
                return self._get_frame(), 1, False, {'event': 'correct_click'}
            else:
                # Wrong click — game over
                return self._get_frame(), -1, True, {'event': 'wrong_click'}
        else:
            # WAIT — always safe
            self.step_in_life += 1
            if self.step_in_life >= self.life_steps:
                return self._get_frame(), 0, True, {'event': 'timeout'}
            return self._get_frame(), 0, False, {'event': 'wait'}


def centered_enc(frame, fold):
    t = F.normalize(torch.from_numpy(frame), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V.mean(dim=0).cpu()
    return t


def main():
    t0 = time.time()
    print("Step 370 -- Local timing game: process() vs temporal patterns", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Flash: 3/20 steps visible. CLICK=level, WAIT=safe.", flush=True)
    print(flush=True)

    fold = CompressedFold(d=D_ENC, k=3)
    game = TimingGame()

    # Seed both classes
    frame = game.reset()
    fold._force_add(centered_enc(frame, fold), label=WAIT)
    frame, _, _, _ = game.step(WAIT)
    fold._force_add(centered_enc(frame, fold), label=CLICK)
    game.reset()

    total_steps = 0
    total_lives = 0
    correct_clicks = 0
    wrong_clicks = 0
    waits = 0
    timeouts = 0
    points = 0

    # Per-bin tracking
    bin_size = 200
    bin_correct = 0
    bin_wrong = 0
    bin_waits = 0
    bins_data = []

    max_steps = 2000

    frame = game.reset()
    total_lives += 1

    while total_steps < max_steps:
        enc = centered_enc(frame, fold)
        action = fold.process_novelty(enc, n_cls=N_ACTS)

        frame, reward, done, info = game.step(action)
        total_steps += 1

        if action == CLICK:
            if info['event'] == 'correct_click':
                correct_clicks += 1
                bin_correct += 1
                points += 1
            elif info['event'] == 'wrong_click':
                wrong_clicks += 1
                bin_wrong += 1
        else:
            waits += 1
            bin_waits += 1
            if info['event'] == 'timeout':
                timeouts += 1

        if done:
            frame = game.reset()
            total_lives += 1

        if total_steps % bin_size == 0:
            total_bin = bin_correct + bin_wrong + bin_waits
            click_rate = (bin_correct + bin_wrong) / max(total_bin, 1)
            precision = bin_correct / max(bin_correct + bin_wrong, 1)
            bins_data.append((total_steps, bin_correct, bin_wrong, bin_waits, click_rate, precision))
            print(f"    [step {total_steps:5d}] correct={bin_correct}"
                  f"  wrong={bin_wrong}  waits={bin_waits}"
                  f"  click_rate={click_rate:.2%}  precision={precision:.2%}"
                  f"  cb={fold.V.shape[0]}", flush=True)
            bin_correct = bin_wrong = bin_waits = 0

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 370 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"steps={total_steps}  lives={total_lives}", flush=True)
    print(f"correct_clicks={correct_clicks}  wrong_clicks={wrong_clicks}"
          f"  waits={waits}  timeouts={timeouts}", flush=True)
    print(f"points={points}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(flush=True)

    total_clicks = correct_clicks + wrong_clicks
    overall_click_rate = total_clicks / max(total_steps, 1)
    overall_precision = correct_clicks / max(total_clicks, 1) if total_clicks > 0 else 0
    random_precision = 3 / 20  # flash is visible 3/20 of the time
    print(f"Overall click rate: {overall_click_rate:.2%}", flush=True)
    print(f"Overall precision (correct/all clicks): {overall_precision:.2%}", flush=True)
    print(f"Random baseline precision: {random_precision:.2%}", flush=True)
    print(flush=True)

    if overall_precision > random_precision * 1.5 and correct_clicks > 5:
        print("PASS: substrate clicks during flash more than random.", flush=True)
    elif correct_clicks == 0:
        print("KILL: substrate never clicks correctly.", flush=True)
    else:
        print("KILL: substrate clicks at random (no timing awareness).", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
