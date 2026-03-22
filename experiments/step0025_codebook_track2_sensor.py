#!/usr/bin/env python3
"""
Track 2: FluxCore CompressedKernel on synthetic ECG sensor data.
Phase 7 domain validation — sensor/physiological signals.

Generates realistic synthetic ECG at 360 Hz with 4 rhythm classes:
  - Normal sinus rhythm (NSR)
  - Tachycardia (fast HR)
  - Bradycardia (slow HR)
  - Arrhythmia (irregular intervals + ectopic beats)

Sliding window of 128 samples -> d=128 feature vector
(64 raw statistical features + 64 FFT magnitude features).
"""

import sys
import math
import random
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v2 import CompressedKernel


# --- Synthetic ECG generation ---

def qrs_template(t, amp=1.0):
    """Simple QRS-like waveform centered at t=0."""
    # P wave
    p = 0.15 * np.exp(-((t + 0.16) ** 2) / (2 * 0.01 ** 2))
    # QRS complex
    q = -0.1 * np.exp(-((t + 0.04) ** 2) / (2 * 0.002 ** 2))
    r = amp * np.exp(-(t ** 2) / (2 * 0.004 ** 2))
    s = -0.15 * np.exp(-((t - 0.04) ** 2) / (2 * 0.002 ** 2))
    # T wave
    tw = 0.25 * np.exp(-((t - 0.2) ** 2) / (2 * 0.02 ** 2))
    return p + q + r + s + tw


def generate_ecg_segment(duration_s, fs=360, rhythm='nsr', seed=None):
    """Generate a synthetic ECG segment.

    rhythm: 'nsr' (60-80 bpm), 'tachy' (100-140 bpm),
            'brady' (40-55 bpm), 'arrhythmia' (irregular)
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    # Set heart rate parameters by rhythm
    if rhythm == 'nsr':
        mean_rr = 0.85  # ~70 bpm
        rr_std = 0.03
        amp = 1.0
    elif rhythm == 'tachy':
        mean_rr = 0.5   # ~120 bpm
        rr_std = 0.02
        amp = 0.8
    elif rhythm == 'brady':
        mean_rr = 1.3   # ~46 bpm
        rr_std = 0.05
        amp = 1.2
    elif rhythm == 'arrhythmia':
        mean_rr = 0.75
        rr_std = 0.25   # high variability
        amp = 1.0
    else:
        raise ValueError(f"Unknown rhythm: {rhythm}")

    # Place beats
    beat_time = 0.0
    beat_times = []
    while beat_time < duration_s:
        beat_times.append(beat_time)
        rr = max(0.3, rng.normal(mean_rr, rr_std))
        beat_time += rr

    # Render each beat
    t_grid = np.arange(n_samples) / fs
    for bt in beat_times:
        local_t = t_grid - bt
        mask = (local_t > -0.3) & (local_t < 0.4)
        beat_amp = amp * rng.uniform(0.85, 1.15) if rhythm == 'arrhythmia' else amp
        signal[mask] += qrs_template(local_t[mask], beat_amp)

    # Add baseline wander + noise
    signal += 0.05 * np.sin(2 * np.pi * 0.15 * t_grid)
    signal += rng.normal(0, 0.02, n_samples)

    return signal


def generate_dataset(n_segments=12, segment_duration=10.0, fs=360, seed=42):
    """Generate multi-rhythm ECG dataset.
    Returns (signal_array, labels_per_window) after windowing.
    """
    rng = np.random.RandomState(seed)
    rhythms = ['nsr', 'tachy', 'brady', 'arrhythmia']

    all_signals = []
    all_labels = []

    for i in range(n_segments):
        rhythm = rhythms[i % len(rhythms)]
        seg = generate_ecg_segment(
            segment_duration, fs=fs, rhythm=rhythm,
            seed=seed + i
        )
        all_signals.append(seg)
        all_labels.extend([rhythm] * len(seg))

    full_signal = np.concatenate(all_signals)
    return full_signal, all_labels, rhythms


def extract_features(window, fs=360):
    """Extract d=128 feature vector from a window of raw samples.
    64 statistical features + 64 FFT magnitude features.
    """
    n = len(window)

    # Statistical features (64): subsample raw + basic stats
    # Downsample to 60 values, then add 4 stats
    indices = np.linspace(0, n - 1, 60, dtype=int)
    raw_sub = window[indices]
    stats = [
        np.mean(window),
        np.std(window),
        np.max(window) - np.min(window),  # range
        np.sum(np.abs(np.diff(window))) / n,  # line length / activity
    ]
    stat_features = np.concatenate([raw_sub, stats])

    # FFT features (64): magnitude spectrum, first 64 bins
    fft_vals = np.abs(np.fft.rfft(window))
    fft_features = fft_vals[:64]
    # Normalize
    fft_norm = np.linalg.norm(fft_features)
    if fft_norm > 1e-10:
        fft_features = fft_features / fft_norm

    features = np.concatenate([stat_features, fft_features])
    # Normalize full vector
    norm = np.linalg.norm(features)
    if norm > 1e-10:
        features = features / norm

    return features.tolist()


def create_stream(signal, window_size=128, step_size=64, fs=360):
    """Create sliding-window feature stream from raw signal."""
    vectors = []
    n = len(signal)
    for start in range(0, n - window_size, step_size):
        window = signal[start:start + window_size]
        vec = extract_features(window, fs)
        vectors.append(vec)
    return vectors


# --- Main experiment ---

def run_experiment():
    print("=" * 70)
    print("  Track 2: FluxCore CompressedKernel on Synthetic ECG Data")
    print("=" * 70)

    # Generate data
    fs = 360
    n_segments = 12  # 3 per rhythm class
    seg_duration = 10.0  # seconds each
    total_duration = n_segments * seg_duration

    print(f"\n  Dataset: Synthetic ECG, {n_segments} segments x {seg_duration}s = {total_duration}s")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Rhythms: NSR, Tachycardia, Bradycardia, Arrhythmia")

    signal, labels, rhythms = generate_dataset(
        n_segments=n_segments, segment_duration=seg_duration, fs=fs
    )
    print(f"  Total samples: {len(signal)}")

    # Create feature stream
    window_size = 128
    step_size = 64
    vectors = create_stream(signal, window_size=window_size, step_size=step_size, fs=fs)
    d = len(vectors[0])
    print(f"  Window: {window_size} samples, step: {step_size}")
    print(f"  Feature vectors: {len(vectors)}, d={d}")

    # Map each vector to its rhythm label
    samples_per_segment = int(seg_duration * fs)
    vec_labels = []
    for i, v in enumerate(vectors):
        sample_center = i * step_size + window_size // 2
        seg_idx = min(sample_center // samples_per_segment, n_segments - 1)
        vec_labels.append(rhythms[seg_idx % len(rhythms)])

    # Initialize kernel
    k = 4
    n_start = 4
    kernel = CompressedKernel(
        n=n_start, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, k_couple=5, max_cells=500
    )

    print(f"\n  Kernel: k={k}, n_start={n_start}, d={d}, tau=0.3, spawning=True, k_couple=5")
    print(f"\n  --- Streaming {len(vectors)} vectors ---\n")

    # Stream data
    total = len(vectors)
    last_dMs = None
    for idx, vec in enumerate(vectors):
        dMs = kernel.step(r=vec)
        last_dMs = dMs

        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            nc = len(kernel.cells)
            ef = kernel.mean_ef_dist()
            au = kernel.mean_autonomy()
            energy = kernel.mean_energy(dMs)
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef_dist={ef:.4f}  autonomy={au:.4f}  energy={energy:.6f}")

    nc_final = len(kernel.cells)
    print(f"\n  --- Final State ---")
    print(f"    Cells: {nc_final} (started {n_start}, spawned {kernel.total_spawned})")
    print(f"    mean_ef_dist:  {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy: {kernel.mean_autonomy():.4f}")
    print(f"    step_count:    {kernel.step_count}")

    # --- Generation survival test (3000 free-running steps, no input) ---
    print(f"\n  --- Generation Survival Test (3000 steps, no input) ---\n")

    cells_before = len(kernel.cells)
    for i in range(3000):
        dMs = kernel.step(r=None)
        if (i + 1) % 500 == 0:
            nc = len(kernel.cells)
            ef = kernel.mean_ef_dist()
            au = kernel.mean_autonomy()
            energy = kernel.mean_energy(dMs)
            print(f"  [gen {i+1:4d}]  cells={nc:3d}  ef_dist={ef:.4f}  autonomy={au:.4f}  energy={energy:.6f}")

    cells_after = len(kernel.cells)
    final_energy = kernel.mean_energy(dMs)
    print(f"\n  --- Generation Survival Results ---")
    print(f"    Cells before: {cells_before}")
    print(f"    Cells after:  {cells_after}")
    print(f"    Final energy: {final_energy:.6f}")
    print(f"    Survived: {'YES' if cells_after >= kernel.n_min and final_energy > 1e-8 else 'NO'}")

    # --- Per-rhythm cell alignment analysis ---
    print(f"\n  --- Rhythm Alignment Analysis ---\n")

    # Compute centroid for each rhythm class
    rhythm_vecs = {r: [] for r in rhythms}
    for vec, label in zip(vectors, vec_labels):
        rhythm_vecs[label].append(vec)

    rhythm_centroids = {}
    for r in rhythms:
        arr = np.array(rhythm_vecs[r])
        centroid = arr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        rhythm_centroids[r] = centroid.tolist()

    # Project centroids
    P = kernel.P
    k = kernel.k
    proj_centroids = {}
    for r, cvec in rhythm_centroids.items():
        d_len = len(cvec)
        flat = [sum(P[i][j] * cvec[j] for j in range(d_len)) for i in range(k * k)]
        proj_centroids[r] = [flat[i * k:(i + 1) * k] for i in range(k)]

    # For each cell, find best rhythm alignment
    print(f"  {'Cell':>4s}  {'Best Rhythm':>12s}  {'Alignment':>10s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")

    covered = set()
    for i, cell in enumerate(kernel.cells):
        best_r = None
        best_a = 0.0
        for r in rhythms:
            a = mcosine(cell, proj_centroids[r])
            if a == a and abs(a) > abs(best_a):  # NaN check
                best_a = a
                best_r = r
        if best_r is not None:
            covered.add(best_r)
        r_label = best_r if best_r is not None else 'N/A'
        print(f"  {i:4d}  {r_label:>12s}  {best_a:>10.4f}")

    print(f"\n  Rhythm coverage: {len(covered)}/{len(rhythms)} ({sorted(covered)})")

    print(f"\n{'='*70}")
    print(f"  TRACK 2 COMPLETE")
    print(f"{'='*70}")

    return {
        'cells_final': nc_final,
        'cells_survived': cells_after,
        'final_energy': final_energy,
        'coverage': len(covered),
        'total_rhythms': len(rhythms),
    }


if __name__ == '__main__':
    results = run_experiment()
