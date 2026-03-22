#!/usr/bin/env python3
"""
Step 517 -- BMR-inspired consolidation on chain centroids.

After dynamic growth chain (CIFAR -> LS20), centroids explode to ~10K.
Bayesian Model Reduction (Heins et al. 2025, AXIOM) merges redundant
model components post-hoc. Here: agglomerative merge of centroids within
L2 distance threshold. NOT cosine -- no codebook ban violation.

Protocol:
  Phase 1: CIFAR 1-pass with dynamic growth (threshold=0.3) -> ~10K centroids
  Measure: NMI_pre, centroid count
  Phase 2: BMR consolidation -- agglomerative clustering on centroids,
           merge centroids within merge_threshold. Reassign edges.
  Measure: NMI_post, centroid count post-merge
  Phase 3: LS20 navigation with consolidated state (5-min)
  Phase 4: CIFAR 1-pass with consolidated state
  Measure: NMI_final, navigation result

Sweep merge_threshold = [0.5, 1.0, 2.0, 3.0] to find Pareto front.

Kill criterion: NMI drops after consolidation at ALL thresholds.
R3 tension: BMR merges = deletion. U3 says growth-only. Is merging
"deletion" or "refinement"? The paper says BMR is principled model
comparison, not arbitrary deletion.

T1: structural test on synthetic data (<30s)
T2: full chain + consolidation run (5-min cap)
"""
import time
import numpy as np
import logging
logging.getLogger().setLevel(logging.WARNING)

SPAWN_THRESHOLD = 0.3
MERGE_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]
N_ARC_ACTIONS_LS20 = 4
TIME_CAP_ARC = 270  # 4.5 min


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class DynamicGraph:
    """Spawn-on-novelty centroid growth. Separate edges per mode."""

    def __init__(self, threshold=SPAWN_THRESHOLD):
        self.threshold = threshold
        self.centroids = None
        self.edges = {}   # mode -> {(cell, action): {next_cell: count}}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._mode = None
        self.spawns = {}

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _spawn(self, x):
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1, -1)])
        self.spawns[self._mode] = self.spawns.get(self._mode, 0) + 1
        return len(self.centroids) - 1

    def step(self, x, n_actions):
        if self.centroids is None:
            cell = self._spawn(x)
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nearest = int(np.argmin(dists))
        cell = self._spawn(x) if dists[nearest] > self.threshold else nearest
        self.cells_seen.add(cell)
        edges = self.edges.setdefault(self._mode, {})
        if self.prev_cell is not None and self.prev_action is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(edges.get((cell, a), {}).values()) for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action

    @property
    def n_centroids(self):
        return len(self.centroids) if self.centroids is not None else 0

    def consolidate(self, merge_threshold):
        """BMR-inspired consolidation: agglomerative merge of nearby centroids.
        Returns (n_before, n_after, merge_map) where merge_map[old_id] = new_id."""
        from scipy.cluster.hierarchy import fcluster, linkage
        n_before = self.n_centroids
        if n_before <= 1:
            return n_before, n_before, {i: i for i in range(n_before)}

        # Agglomerative clustering on centroids using L2 distance
        Z = linkage(self.centroids, method='average', metric='euclidean')
        labels = fcluster(Z, t=merge_threshold, criterion='distance')
        labels = labels - 1  # 0-indexed

        n_after = len(set(labels))
        merge_map = {}
        new_centroids = []
        cluster_ids = sorted(set(labels))
        cluster_to_new = {c: i for i, c in enumerate(cluster_ids)}

        for c in cluster_ids:
            mask = labels == c
            # New centroid = mean of merged centroids
            new_centroids.append(self.centroids[mask].mean(axis=0))

        for old_id in range(n_before):
            merge_map[old_id] = cluster_to_new[labels[old_id]]

        self.centroids = np.array(new_centroids, dtype=np.float32)

        # Remap edges
        new_edges = {}
        for mode, mode_edges in self.edges.items():
            new_mode_edges = {}
            for (cell, action), transitions in mode_edges.items():
                new_cell = merge_map.get(cell, cell)
                new_key = (new_cell, action)
                if new_key not in new_mode_edges:
                    new_mode_edges[new_key] = {}
                for next_cell, count in transitions.items():
                    new_next = merge_map.get(next_cell, next_cell)
                    new_mode_edges[new_key][new_next] = (
                        new_mode_edges[new_key].get(new_next, 0) + count
                    )
            new_edges[mode] = new_mode_edges
        self.edges = new_edges

        # Remap cells_seen
        self.cells_seen = {merge_map.get(c, c) for c in self.cells_seen}

        return n_before, n_after, merge_map


def cifar_nmi(centroids, X_enc, labels):
    """Assign each image to nearest centroid, compute NMI."""
    from sklearn.metrics import normalized_mutual_info_score
    assignments = np.zeros(len(X_enc), dtype=np.int32)
    for i in range(len(X_enc)):
        diffs = centroids - X_enc[i]
        dists = np.sum(diffs * diffs, axis=1)
        assignments[i] = int(np.argmin(dists))
    nmi = normalized_mutual_info_score(labels, assignments, average_method='arithmetic')
    # Purity
    n_c = len(centroids)
    purities = []
    for c in range(n_c):
        mask = assignments == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels[mask], minlength=100)
        purities.append(counts.max() / mask.sum())
    purity = float(np.mean(purities)) if purities else 0.0
    return nmi, purity


def run_ls20(g, arc, game_id, duration=TIME_CAP_ARC):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while time.time() - t0 < duration:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS_LS20)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  LS20: {status}  centroids={g.n_centroids}  go={go}  steps={ts}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


# ============================================================
# T1: Structural test (synthetic data)
# ============================================================

def run_t1():
    print("T1: Structural test (synthetic BMR consolidation)", flush=True)
    t0 = time.time()

    # Build a graph with 200 synthetic centroids in 4 dense clusters
    g = DynamicGraph(threshold=0.01)  # very tight -> many spawns
    g.set_mode('test')
    rng = np.random.RandomState(42)

    # 4 cluster centers in 256D, well separated
    # Within-cluster L2 ~ sqrt(256)*0.05*sqrt(2) ~ 1.13
    # Between-cluster L2 ~ sqrt(256)*10*sqrt(2) ~ 226
    # merge_threshold=3.0 should merge within-cluster, keep between-cluster
    centers = rng.randn(4, 256).astype(np.float32) * 10.0
    for c in range(4):
        for _ in range(50):
            x = centers[c] + rng.randn(256).astype(np.float32) * 0.05
            g.step(x, 4)
    n_pre = g.n_centroids
    assert n_pre >= 100, f"Expected >=100 centroids, got {n_pre}"

    # Consolidate at threshold that should merge within-cluster
    n_before, n_after, merge_map = g.consolidate(merge_threshold=3.0)
    assert n_after < n_before, f"Consolidation did nothing: {n_before} -> {n_after}"
    assert n_after >= 4, f"Over-merged: {n_after} < 4 expected clusters"
    assert len(merge_map) == n_before, f"merge_map incomplete"

    # Verify edges were remapped
    test_edges = g.edges.get('test', {})
    for (cell, action), transitions in test_edges.items():
        assert cell < n_after, f"Edge cell {cell} >= {n_after}"
        for next_cell in transitions:
            assert next_cell < n_after, f"Edge target {next_cell} >= {n_after}"

    # Verify centroids array matches
    assert len(g.centroids) == n_after

    # Verify navigation still works after consolidation
    x = centers[0] + rng.randn(256).astype(np.float32) * 0.1
    a = g.step(x, 4)
    assert 0 <= a < 4

    print(f"  PASS: {n_before} -> {n_after} centroids. Edges remapped. Navigation works. "
          f"{time.time()-t0:.1f}s", flush=True)
    return True


# ============================================================
# T2: Full chain + consolidation sweep
# ============================================================

def run_t2():
    print("\nLoading data...", flush=True)
    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    print(f"  CIFAR-100: {len(X)} images, {len(set(y))} classes", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # Pre-encode CIFAR for NMI measurements
    print("  Encoding CIFAR...", flush=True)
    X_enc = np.zeros((len(X), 256), dtype=np.float32)
    for i in range(len(X)):
        X_enc[i] = encode_cifar(X[i])

    results = {}
    for merge_th in MERGE_THRESHOLDS:
        print(f"\n{'='*60}", flush=True)
        print(f"MERGE THRESHOLD = {merge_th}", flush=True)
        print(f"{'='*60}", flush=True)

        np.random.seed(0)
        g = DynamicGraph(threshold=SPAWN_THRESHOLD)

        # Phase 1: CIFAR 1-pass -> spawn centroids
        print("\n--- Phase 1: CIFAR (1-pass, spawn) ---", flush=True)
        g.set_mode('cifar')
        t0 = time.time()
        for i in range(len(X)):
            x = encode_cifar(X[i])
            g.step(x, 100)
        print(f"  centroids={g.n_centroids}  spawns={g.spawns}  {time.time()-t0:.0f}s", flush=True)

        # Measure NMI pre-consolidation
        nmi_pre, pur_pre = cifar_nmi(g.centroids, X_enc, y)
        print(f"  NMI_pre={nmi_pre:.4f}  purity_pre={pur_pre:.4f}", flush=True)
        n_pre = g.n_centroids

        # Phase 2: BMR consolidation
        print(f"\n--- Phase 2: BMR consolidation (merge_threshold={merge_th}) ---", flush=True)
        t0 = time.time()
        n_before, n_after, merge_map = g.consolidate(merge_threshold=merge_th)
        elapsed_merge = time.time() - t0
        print(f"  {n_before} -> {n_after} centroids ({n_before - n_after} merged)  "
              f"{elapsed_merge:.1f}s", flush=True)

        # Measure NMI post-consolidation
        nmi_post, pur_post = cifar_nmi(g.centroids, X_enc, y)
        print(f"  NMI_post={nmi_post:.4f}  purity_post={pur_post:.4f}", flush=True)
        nmi_delta = nmi_post - nmi_pre
        print(f"  NMI delta: {nmi_delta:+.4f}", flush=True)

        # Phase 3: LS20 navigation with consolidated state
        print(f"\n--- Phase 3: LS20 (consolidated, {TIME_CAP_ARC}s cap) ---", flush=True)
        lvls, level_step = run_ls20(g, arc, ls20.game_id, TIME_CAP_ARC)

        # Phase 4: CIFAR 1-pass with consolidated state
        print(f"\n--- Phase 4: CIFAR (1-pass, post-chain) ---", flush=True)
        g.set_mode('cifar')
        correct = 0
        t0 = time.time()
        for i in range(len(X)):
            x = encode_cifar(X[i])
            a = g.step(x, 100)
            if a == int(y[i]):
                correct += 1
        acc_final = correct / len(X) * 100
        n_final = g.n_centroids
        nmi_final, pur_final = cifar_nmi(g.centroids, X_enc, y)
        print(f"  acc={acc_final:.2f}%  centroids={n_final}  "
              f"NMI={nmi_final:.4f}  purity={pur_final:.4f}  {time.time()-t0:.0f}s", flush=True)

        results[merge_th] = {
            'n_pre': n_pre, 'n_post': n_after, 'n_final': n_final,
            'nmi_pre': nmi_pre, 'nmi_post': nmi_post, 'nmi_final': nmi_final,
            'pur_pre': pur_pre, 'pur_post': pur_post, 'pur_final': pur_final,
            'nmi_delta': nmi_delta,
            'ls20_lvls': lvls, 'ls20_step': level_step,
            'acc_final': acc_final, 'merge_time': elapsed_merge,
        }

    return results


def main():
    t_total = time.time()
    print("Step 517: BMR-inspired consolidation on chain centroids", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}  merge_thresholds={MERGE_THRESHOLDS}", flush=True)
    print(f"Literature: Heins et al. 2025 (AXIOM) -- BMR for model reduction", flush=True)
    print(f"R3 tension: U3 (growth-only) vs BMR (merge = principled deletion)", flush=True)

    # T1
    if not run_t1():
        print("T1 FAILED -- aborting"); return

    # T2
    results = run_t2()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("STEP 517 SUMMARY", flush=True)
    print(f"  {'merge_th':>8}  {'n_pre':>6}  {'n_post':>7}  {'n_final':>8}  "
          f"{'NMI_pre':>8}  {'NMI_post':>9}  {'NMI_delta':>10}  {'LS20':>10}  "
          f"{'merge_s':>8}", flush=True)
    for th in MERGE_THRESHOLDS:
        r = results[th]
        ls20_str = f"WIN@{r['ls20_step']}" if r['ls20_lvls'] > 0 else "FAIL"
        print(f"  {th:>8.1f}  {r['n_pre']:>6}  {r['n_post']:>7}  {r['n_final']:>8}  "
              f"{r['nmi_pre']:>8.4f}  {r['nmi_post']:>9.4f}  {r['nmi_delta']:>+10.4f}  "
              f"{ls20_str:>10}  {r['merge_time']:>7.1f}s", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    any_nmi_up = any(r['nmi_delta'] > 0 for r in results.values())
    any_nav = any(r['ls20_lvls'] > 0 for r in results.values())
    best_th = max(results.keys(), key=lambda t: results[t]['nmi_post'])
    best = results[best_th]

    if any_nmi_up and any_nav:
        print(f"  BMR WORKS: merge_threshold={best_th} gives NMI={best['nmi_post']:.4f} "
              f"(delta={best['nmi_delta']:+.4f}) with {best['n_post']} centroids "
              f"(was {best['n_pre']}).", flush=True)
        nav_ths = [th for th in MERGE_THRESHOLDS if results[th]['ls20_lvls'] > 0]
        print(f"  Navigation survives at merge_threshold={nav_ths}.", flush=True)
        print(f"  Pareto: fewer centroids, equal/better NMI, navigation preserved.", flush=True)
        print(f"\n  R3 TENSION RESOLVED: BMR is not arbitrary deletion. It is evidence-based", flush=True)
        print(f"  model comparison. U3 (growth-only) holds during ONLINE learning.", flush=True)
        print(f"  BMR is OFFLINE refinement -- a separate phase, not a change to process().", flush=True)
    elif any_nmi_up and not any_nav:
        print(f"  PARTIAL: NMI improves (best delta={best['nmi_delta']:+.4f}) but navigation", flush=True)
        print(f"  breaks at ALL thresholds. Consolidation helps classification, kills navigation.", flush=True)
        print(f"  U3 is load-bearing for navigation: growth-only = edge consistency.", flush=True)
    elif not any_nmi_up and any_nav:
        print(f"  NMI DROPS at all thresholds. Consolidation hurts classification.", flush=True)
        print(f"  Navigation survives. BMR consolidation is pure information loss here.", flush=True)
    else:
        print(f"  KILL: NMI drops AND navigation breaks. BMR consolidation is destructive.", flush=True)
        print(f"  U3 (growth-only) is validated -- merging IS deletion, not refinement.", flush=True)

    # Key numbers for paper
    print(f"\n  Centroid reduction ratios:", flush=True)
    for th in MERGE_THRESHOLDS:
        r = results[th]
        ratio = r['n_post'] / r['n_pre'] * 100
        print(f"    merge_th={th}: {r['n_pre']} -> {r['n_post']} ({ratio:.1f}% retained)", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
