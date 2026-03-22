"""
Step 595 -- Chain tax measurement.

Does CIFAR data contaminate LS20 navigation via a shared LSH substrate?
Does LS20 navigation affect the substrate's CIFAR class structure?

Three conditions, 5 seeds each:
  1. LS20 alone (baseline time-to-L1, clean substrate)
  2. CIFAR(1K) -> LS20(10K)  (contamination: CIFAR edges in shared G before LS20)
  3. CIFAR(1K) -> LS20(10K) -> CIFAR NMI  (class structure preserved?)

Design:
  - SHARED H (same random hyperplanes for both domains)
  - SHARED G (same edge dict for both domains -- intentional contamination test)
  - Per-frame centering (x -= x.mean()) for both CIFAR and LS20
  - CIFAR "actions": random from {0,1,2,3} (no meaningful action space)
  - NMI: hash test images -> cells -> NMI(cells, true_labels)

Expected:
  - If CIFAR and LS20 hash to DIFFERENT cells: chain_tax = 0 (domain isolation)
  - If they share cells: CIFAR edges pollute argmin, time-to-L1 increases
  - NMI_P1 == NMI_P3 (hash deterministic, G doesn't affect hash)
  - If NMI > 0: encoding has class signal

Fits 5-min cap (5 seeds x ~33K steps total).
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
N_CIFAR = 1000         # CIFAR samples per phase
MAX_LS20_STEPS = 10_000
TIME_CAP_LS20 = 60     # seconds per LS20 seed
N_SEEDS = 5


# ── encoding ──────────────────────────────────────────────────────────────────

def enc_ls20(frame):
    """LS20 frame[0] 64x64 [0-15] -> avgpool4 -> 16x16 = 256D, centered."""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_cifar(img):
    """CIFAR (32,32,3) uint8 -> grayscale -> avgpool2 -> 16x16 = 256D, centered."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)   # CHW -> HWC
    gray = img.mean(axis=2).astype(np.float32) / 255.0
    x = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


# ── substrate ─────────────────────────────────────────────────────────────────

class SharedLSH:
    """LSH K=12 with SHARED G (intentional contamination test)."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.ls20_cells = set()
        self.cifar_cells = set()
        self.total_deaths = 0

    def observe_ls20(self, frame):
        x = enc_ls20(frame)
        n = lsh_hash(x, self.H)
        self.ls20_cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def observe_cifar(self, img):
        x = enc_cifar(img)
        n = lsh_hash(x, self.H)
        self.cifar_cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        self.total_deaths += 1

    def on_reset(self):
        self._pn = None

    def cell_overlap(self):
        return len(self.ls20_cells & self.cifar_cells)


# ── CIFAR loading ─────────────────────────────────────────────────────────────

def load_cifar100(n=N_CIFAR):
    """Load N CIFAR-100 test images. Returns (imgs, labels) or None."""
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T
        import tempfile, os
        ds = dsets.CIFAR100(root=os.path.join(tempfile.gettempdir(), 'cifar100'),
                            train=False, download=True,
                            transform=T.ToTensor())
        imgs, labels = [], []
        for i in range(min(n, len(ds))):
            img, label = ds[i]
            imgs.append((img.numpy() * 255).astype(np.uint8))
            labels.append(label)
        print(f"  CIFAR-100: loaded {len(imgs)} test images", flush=True)
        return np.array(imgs), np.array(labels)
    except Exception as e:
        print(f"  CIFAR-100 unavailable ({e}). Using synthetic.", flush=True)
        # Synthetic: random 32x32x3 images with 100 pseudo-classes
        rng = np.random.RandomState(42)
        imgs = rng.randint(0, 256, (n, 3, 32, 32), dtype=np.uint8)
        labels = rng.randint(0, 100, n)
        return imgs, labels


def compute_nmi(imgs, labels, H):
    """Hash images, compute NMI(cell, label)."""
    cells = np.array([lsh_hash(enc_cifar(img), H) for img in imgs])
    try:
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(labels, cells)
        return float(nmi), cells
    except ImportError:
        # Manual NMI approximation
        return float('nan'), cells


def cifar_contaminate(sub, imgs, labels, rng, n_steps):
    """Process CIFAR images through shared substrate (builds G with CIFAR edges)."""
    sub.on_reset()
    for i in range(min(n_steps, len(imgs))):
        sub.observe_cifar(imgs[i])
        _ = sub.act()   # choose action (argmin over current G)
    cifar_cell_count = len(sub.cifar_cells)
    return cifar_cell_count


# ── LS20 runner ───────────────────────────────────────────────────────────────

def run_ls20(mk, seed, sub):
    env = mk()
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = go = step = 0
    prev_cl = 0
    fresh = True
    t0 = time.time()
    l1_step = None

    while step < MAX_LS20_STEPS and time.time() - t0 < TIME_CAP_LS20:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        sub.observe_ls20(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1
        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1_step is None:
                l1_step = step
        prev_cl = cl

    return l1, l1_step, go, step, len(sub.ls20_cells)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 595: Chain tax measurement", flush=True)
    print(f"  K={K} | N_CIFAR={N_CIFAR} | {N_SEEDS} seeds | LS20={MAX_LS20_STEPS} steps", flush=True)

    # Load CIFAR-100 once (same for all seeds)
    cifar_imgs, cifar_labels = load_cifar100(N_CIFAR)
    rng = np.random.RandomState(0)

    t_total = time.time()

    # Results
    c1_l1_steps = []   # LS20 alone: step when first L1 found (None if not found)
    c2_l1_steps = []   # CIFAR->LS20: step when first L1 found
    c3_nmi_p1 = []     # CIFAR NMI before LS20
    c3_nmi_p3 = []     # CIFAR NMI after LS20
    c1_wins = c2_wins = 0
    overlaps = []

    for seed in range(N_SEEDS):
        print(f"\n--- seed {seed} ---", flush=True)

        # Condition 1: LS20 alone (clean G)
        sub1 = SharedLSH(seed=seed * 100 + 7)
        l1, l1_step, go, steps, cells = run_ls20(mk, seed, sub1)
        c1_wins += (l1 > 0)
        c1_l1_steps.append(l1_step)
        print(f"  C1 (LS20 alone):    L1={l1} first@{l1_step} go={go} cells={cells}", flush=True)

        # Condition 2: CIFAR -> LS20 (contaminated G)
        sub2 = SharedLSH(seed=seed * 100 + 7)
        cifar_cells_c2 = cifar_contaminate(sub2, cifar_imgs, cifar_labels, rng, N_CIFAR)
        overlap = sub2.cell_overlap()
        # Note: cell_overlap computed after CIFAR phase, before LS20
        # ls20_cells is empty here — overlap only shows after LS20
        l1, l1_step, go, steps, ls20_cells = run_ls20(mk, seed, sub2)
        c2_wins += (l1 > 0)
        c2_l1_steps.append(l1_step)
        post_overlap = sub2.cell_overlap()
        overlaps.append(post_overlap)
        print(f"  C2 (CIFAR->LS20):   L1={l1} first@{l1_step} go={go} cells={ls20_cells} "
              f"cifar_cells={cifar_cells_c2} overlap={post_overlap}", flush=True)

        # Condition 3: CIFAR -> LS20 -> CIFAR NMI
        sub3 = SharedLSH(seed=seed * 100 + 7)
        # P1 NMI (before LS20)
        nmi_p1, cells_p1 = compute_nmi(cifar_imgs, cifar_labels, sub3.H)
        # Run CIFAR contamination
        cifar_contaminate(sub3, cifar_imgs, cifar_labels, rng, N_CIFAR)
        # Run LS20
        run_ls20(mk, seed, sub3)
        # P3 NMI (after LS20 — hash assignments are deterministic, should be same)
        nmi_p3, cells_p3 = compute_nmi(cifar_imgs, cifar_labels, sub3.H)
        c3_nmi_p1.append(nmi_p1)
        c3_nmi_p3.append(nmi_p3)
        print(f"  C3 (chain NMI):     NMI_P1={nmi_p1:.4f} NMI_P3={nmi_p3:.4f} "
              f"delta={nmi_p3-nmi_p1:.4f}", flush=True)

    # Summary
    def fmt_steps(steps):
        valid = [s for s in steps if s is not None]
        if not valid:
            return "never"
        return f"avg={np.mean(valid):.0f} (n={len(valid)}/{N_SEEDS})"

    avg_overlap = np.mean(overlaps) if overlaps else 0
    avg_nmi_p1 = np.nanmean(c3_nmi_p1) if c3_nmi_p1 else float('nan')
    avg_nmi_p3 = np.nanmean(c3_nmi_p3) if c3_nmi_p3 else float('nan')

    print(f"\n{'='*60}", flush=True)
    print(f"Step 595: Chain tax measurement ({N_SEEDS} seeds)", flush=True)
    print(f"\n  C1 LS20 alone:    {c1_wins}/{N_SEEDS} L1  time-to-L1: {fmt_steps(c1_l1_steps)}", flush=True)
    print(f"  C2 CIFAR->LS20:   {c2_wins}/{N_SEEDS} L1  time-to-L1: {fmt_steps(c2_l1_steps)}", flush=True)

    # Time-to-L1 change
    c1_valid = [s for s in c1_l1_steps if s is not None]
    c2_valid = [s for s in c2_l1_steps if s is not None]
    if c1_valid and c2_valid:
        tax = np.mean(c2_valid) - np.mean(c1_valid)
        print(f"  Chain tax (L1 step delta): {tax:+.0f} steps "
              f"({'slower' if tax > 0 else 'faster' if tax < 0 else 'same'})", flush=True)
    else:
        print(f"  Chain tax: insufficient L1 data", flush=True)

    print(f"\n  Cell overlap (CIFAR/LS20 shared cells): avg={avg_overlap:.1f}", flush=True)
    if avg_overlap == 0:
        print(f"  DOMAIN ISOLATED: CIFAR and LS20 hash to completely separate cells.", flush=True)
    else:
        print(f"  DOMAIN OVERLAP: {avg_overlap:.1f} shared cells cause cross-domain contamination.", flush=True)

    print(f"\n  C3 NMI: P1={avg_nmi_p1:.4f} P3={avg_nmi_p3:.4f} "
          f"delta={avg_nmi_p3-avg_nmi_p1:.4f}", flush=True)
    if abs(avg_nmi_p3 - avg_nmi_p1) < 0.005:
        print(f"  NMI STABLE: LS20 navigation does not affect CIFAR class structure.", flush=True)
        print(f"  (Expected: hash assignments are deterministic given H.)", flush=True)
    else:
        print(f"  NMI SHIFTED: unexpected change — investigate.", flush=True)

    if avg_nmi_p1 > 0.1:
        print(f"\n  ENCODING HAS CLASS SIGNAL: NMI={avg_nmi_p1:.4f} > 0.1.", flush=True)
        print(f"  Consistent with Step 526 (NMI=0.48 at k=12).", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
