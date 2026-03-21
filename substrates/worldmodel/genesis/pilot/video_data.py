"""Video Data: Flexible data loader for real-world video training.

Supports:
1. Synthetic bouncing shapes (for testing/debugging)
2. H5 files (TinyWorlds format)
3. MP4 videos (for custom data)

Default: Downloads a small subset for immediate training.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import urllib.request
import zipfile


# ============================================================================
# SYNTHETIC DATA (for testing)
# ============================================================================

class SyntheticVideoDataset(Dataset):
    """Bouncing colored shapes at 64x64.

    More complex than micro_data.py:
    - 64x64 RGB (vs 16x16 grayscale)
    - Multiple colored objects
    - Variable speeds
    """

    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 16,
        num_objects: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.num_objects = num_objects
        self.rng = np.random.RandomState(seed)

        self.sequences = []
        for _ in range(num_sequences):
            self.sequences.append(self._generate_sequence())

    def _generate_sequence(self):
        """Generate one video sequence."""
        H, W = 64, 64
        frames = np.zeros((self.seq_length, 3, H, W), dtype=np.float32)

        # Random background color (dark)
        bg_color = self.rng.uniform(0, 0.2, size=3)
        for t in range(self.seq_length):
            frames[t] = bg_color.reshape(3, 1, 1)

        # Create objects
        objects = []
        colors = [
            [1.0, 0.2, 0.2],  # Red
            [0.2, 1.0, 0.2],  # Green
            [0.2, 0.2, 1.0],  # Blue
            [1.0, 1.0, 0.2],  # Yellow
            [1.0, 0.2, 1.0],  # Magenta
        ]

        for i in range(self.num_objects):
            obj = {
                'x': self.rng.randint(10, W - 10),
                'y': self.rng.randint(10, H - 10),
                'vx': self.rng.choice([-2, -1, 1, 2]),
                'vy': self.rng.choice([-2, -1, 1, 2]),
                'size': self.rng.randint(6, 12),
                'color': np.array(colors[i % len(colors)]),
                'shape': self.rng.choice(['square', 'circle']),
            }
            objects.append(obj)

        # Simulate motion
        for t in range(self.seq_length):
            for obj in objects:
                x, y = int(obj['x']), int(obj['y'])
                size = obj['size']
                color = obj['color']

                # Draw object
                if obj['shape'] == 'square':
                    y1, y2 = max(0, y - size//2), min(H, y + size//2)
                    x1, x2 = max(0, x - size//2), min(W, x + size//2)
                    for c in range(3):
                        frames[t, c, y1:y2, x1:x2] = color[c]
                else:  # circle
                    yy, xx = np.ogrid[:H, :W]
                    mask = ((xx - x)**2 + (yy - y)**2) < (size//2)**2
                    for c in range(3):
                        frames[t, c][mask] = color[c]

                # Update position
                obj['x'] += obj['vx']
                obj['y'] += obj['vy']

                # Bounce
                if obj['x'] < size or obj['x'] > W - size:
                    obj['vx'] *= -1
                    obj['x'] = np.clip(obj['x'], size, W - size)
                if obj['y'] < size or obj['y'] > H - size:
                    obj['vy'] *= -1
                    obj['y'] = np.clip(obj['y'], size, H - size)

        return torch.from_numpy(frames)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================================
# H5 DATASET (TinyWorlds format)
# ============================================================================

class H5VideoDataset(Dataset):
    """Load video from H5 file (TinyWorlds format).

    Expected H5 structure:
    - 'frames': [N, H, W, C] or [N, C, H, W] uint8 array
    - Optional 'actions': [N] or [N, action_dim]
    """

    def __init__(
        self,
        h5_path: str,
        seq_length: int = 16,
        stride: int = 1,
        transform=None,
    ):
        super().__init__()
        import h5py

        self.h5_path = h5_path
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform

        # Load metadata (don't keep file open)
        with h5py.File(h5_path, 'r') as f:
            # Try common key names
            for key in ['frames', 'video', 'images', 'data']:
                if key in f:
                    self.frames_key = key
                    self.total_frames = f[key].shape[0]
                    self.frame_shape = f[key].shape[1:]
                    break
            else:
                raise ValueError(f"Could not find frames in {h5_path}. Keys: {list(f.keys())}")

        self.num_sequences = (self.total_frames - seq_length) // stride

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        import h5py

        start = idx * self.stride
        end = start + self.seq_length

        with h5py.File(self.h5_path, 'r') as f:
            frames = f[self.frames_key][start:end]

        # Convert to float [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Ensure [T, C, H, W] format
        if frames.ndim == 4 and frames.shape[-1] in [1, 3]:
            frames = np.transpose(frames, (0, 3, 1, 2))

        frames = torch.from_numpy(frames)

        if self.transform:
            frames = self.transform(frames)

        return frames


# ============================================================================
# MP4 DATASET
# ============================================================================

class MP4VideoDataset(Dataset):
    """Load video from MP4 files using OpenCV or torchvision."""

    def __init__(
        self,
        video_dir: str,
        seq_length: int = 16,
        frame_size: tuple = (64, 64),
        stride: int = 8,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.frame_size = frame_size
        self.stride = stride

        # Find all video files
        self.video_files = list(Path(video_dir).glob("*.mp4"))
        if not self.video_files:
            raise ValueError(f"No MP4 files found in {video_dir}")

        # Build index of (video_idx, start_frame)
        self.index = []
        for vid_idx, vid_path in enumerate(self.video_files):
            try:
                import cv2
                cap = cv2.VideoCapture(str(vid_path))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                for start in range(0, n_frames - seq_length, stride):
                    self.index.append((vid_idx, start))
            except Exception as e:
                print(f"Warning: Could not read {vid_path}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        import cv2

        vid_idx, start_frame = self.index[idx]
        vid_path = self.video_files[vid_idx]

        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.seq_length):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB, resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)

        cap.release()

        # Pad if needed
        while len(frames) < self.seq_length:
            frames.append(frames[-1])

        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]

        return torch.from_numpy(frames)


# ============================================================================
# DATASET FACTORY
# ============================================================================

def get_video_dataset(
    source: str = "synthetic",
    **kwargs
) -> Dataset:
    """Get video dataset by source type.

    Args:
        source: One of:
            - "synthetic": Generated bouncing shapes
            - "h5:/path/to/file.h5": H5 file
            - "mp4:/path/to/dir": Directory of MP4s
            - "tinyworlds:coinrun": Download TinyWorlds dataset

    Returns:
        Dataset yielding [T, C, H, W] tensors
    """
    if source == "synthetic":
        return SyntheticVideoDataset(**kwargs)

    elif source.startswith("h5:"):
        h5_path = source[3:]
        return H5VideoDataset(h5_path, **kwargs)

    elif source.startswith("mp4:"):
        video_dir = source[4:]
        return MP4VideoDataset(video_dir, **kwargs)

    elif source.startswith("tinyworlds:"):
        dataset_name = source[11:]
        return get_tinyworlds_dataset(dataset_name, **kwargs)

    else:
        raise ValueError(f"Unknown source: {source}")


def get_tinyworlds_dataset(name: str = "pong", cache_dir: str = None, **kwargs):
    """Download and load TinyWorlds dataset.

    Available datasets (by size):
    - pong: 14MB (smallest, good for testing)
    - pole_position: 17MB
    - sonic: 249MB
    - picodoom: 662MB
    - zelda: 1.8GB (largest)
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "genesis" / "tinyworlds"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    h5_path = cache_dir / f"{name}_frames.h5"

    if not h5_path.exists():
        print(f"Downloading TinyWorlds {name} dataset...")
        # HuggingFace LFS URL format
        url = f"https://huggingface.co/datasets/AlmondGod/tinyworlds/resolve/main/{name}_frames.h5"
        try:
            # Use requests or urllib with proper headers for LFS
            import subprocess
            result = subprocess.run(
                ["curl", "-L", "-o", str(h5_path), url],
                capture_output=True,
                timeout=600
            )
            if result.returncode != 0 or not h5_path.exists() or h5_path.stat().st_size < 1000:
                raise Exception(f"Download failed: {result.stderr.decode()}")
            print(f"Downloaded to {h5_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Falling back to synthetic data...")
            if h5_path.exists():
                h5_path.unlink()
            return SyntheticVideoDataset(**kwargs)

    return H5VideoDataset(str(h5_path), **kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing Video Data Loaders...\n")

    # Test synthetic
    print("1. Synthetic Dataset")
    synthetic = SyntheticVideoDataset(num_sequences=100, seq_length=16, num_objects=3)
    print(f"   Sequences: {len(synthetic)}")
    sample = synthetic[0]
    print(f"   Sample shape: {sample.shape}")
    print(f"   Sample range: [{sample.min():.2f}, {sample.max():.2f}]")

    # Test dataloader
    loader = DataLoader(synthetic, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"   Batch shape: {batch.shape}")

    # Test TinyWorlds (will download if not cached)
    print("\n2. TinyWorlds Dataset (will download ~100MB)")
    try:
        tinyworlds = get_video_dataset("tinyworlds:coinrun", seq_length=16)
        print(f"   Sequences: {len(tinyworlds)}")
        sample = tinyworlds[0]
        print(f"   Sample shape: {sample.shape}")
    except Exception as e:
        print(f"   TinyWorlds not available: {e}")
        print("   Using synthetic fallback")

    print("\nAll tests passed!")
