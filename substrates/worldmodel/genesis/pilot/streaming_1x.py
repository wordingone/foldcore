"""Streaming Data Loader for 1X World Model Dataset.

Streams data in chunks without loading entire dataset into memory.
Supports continuous world sequences with proper segment handling.

Dataset: https://huggingface.co/datasets/1x-technologies/worldmodel
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Optional, Iterator, Tuple
import subprocess


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================

def download_1x_dataset(
    local_dir: str = "data/1x_worldmodel",
    version: str = "v1.1",
) -> Path:
    """Download 1X World Model dataset from HuggingFace.

    Args:
        local_dir: Local directory to store data
        version: Dataset version (v1.1 or v2.0)

    Returns:
        Path to downloaded data directory
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    marker_file = local_dir / f".downloaded_{version}"

    if marker_file.exists():
        print(f"Dataset already downloaded at {local_dir}")
        return local_dir

    print(f"Downloading 1X World Model dataset ({version})...")
    print("This may take a while (~10GB)...")

    # Use huggingface-cli to download
    cmd = [
        "huggingface-cli", "download",
        "1x-technologies/worldmodel",
        "--repo-type", "dataset",
        "--local-dir", str(local_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")

        marker_file.touch()
        print(f"Downloaded to {local_dir}")

    except FileNotFoundError:
        print("huggingface-cli not found. Install with: pip install huggingface_hub")
        print("Or download manually from: https://huggingface.co/datasets/1x-technologies/worldmodel")
        raise

    return local_dir


# =============================================================================
# MEMORY-MAPPED DATASET (Efficient, doesn't load all into RAM)
# =============================================================================

class MemoryMapped1XDataset(Dataset):
    """Memory-mapped dataset for 1X World Model data.

    Uses np.memmap for efficient streaming - only loads requested chunks.
    Handles segment boundaries for continuous sequences.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 16,
        stride: int = 8,
        include_actions: bool = True,
    ):
        """
        Args:
            data_dir: Path to downloaded 1X data
            seq_length: Number of frames per sequence
            stride: Stride between sequences (for overlap)
            include_actions: Whether to load action data
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.stride = stride
        self.include_actions = include_actions

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            # Try to infer from files
            self.metadata = self._infer_metadata()

        self.num_frames = self.metadata["num_images"]
        self.token_size = self.metadata.get("s", 16)  # 16x16 tokens
        self.token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))

        # Memory-map video tokens
        video_path = self.data_dir / "video.bin"
        self.video_tokens = np.memmap(
            video_path,
            dtype=self.token_dtype,
            mode="r",
            shape=(self.num_frames, self.token_size, self.token_size)
        )

        # Memory-map segment IDs (for detecting sequence boundaries)
        segment_path = self.data_dir / "segment_ids.bin"
        if segment_path.exists():
            self.segment_ids = np.memmap(
                segment_path,
                dtype=np.int32,
                mode="r",
                shape=(self.num_frames,)
            )
        else:
            self.segment_ids = None

        # Memory-map actions if available
        self.actions = None
        if include_actions:
            actions_dir = self.data_dir / "actions"
            if actions_dir.exists():
                self._load_actions(actions_dir)

        # Build valid sequence indices (respecting segment boundaries)
        self.valid_indices = self._build_valid_indices()

        print(f"Loaded 1X dataset: {self.num_frames:,} frames, {len(self.valid_indices):,} sequences")

    def _infer_metadata(self) -> dict:
        """Infer metadata from file sizes."""
        video_path = self.data_dir / "video.bin"
        file_size = video_path.stat().st_size

        # Assume 16x16 tokens, uint32
        token_size = 16
        bytes_per_frame = token_size * token_size * 4  # uint32
        num_frames = file_size // bytes_per_frame

        return {
            "num_images": num_frames,
            "s": token_size,
            "token_dtype": "uint32",
        }

    def _load_actions(self, actions_dir: Path):
        """Load action arrays from actions directory."""
        # 1X uses multiple action files (joint_pos, driving_command, etc.)
        action_files = {
            "joint_pos": (21,),  # 21 joint positions
            "neck_desired": (2,),  # neck pitch/yaw
            "driving_command": (2,),  # linear/angular velocity
        }

        self.actions = {}
        for name, shape in action_files.items():
            path = actions_dir / f"{name}.bin"
            if path.exists():
                full_shape = (self.num_frames,) + shape
                self.actions[name] = np.memmap(
                    path,
                    dtype=np.float32,
                    mode="r",
                    shape=full_shape
                )

    def _build_valid_indices(self) -> list:
        """Build list of valid starting indices for sequences.

        Ensures sequences don't cross segment boundaries.
        """
        valid = []

        for i in range(0, self.num_frames - self.seq_length, self.stride):
            # Check if all frames in sequence belong to same segment
            if self.segment_ids is not None:
                segment_start = self.segment_ids[i]
                segment_end = self.segment_ids[i + self.seq_length - 1]
                if segment_start != segment_end:
                    continue  # Skip - crosses segment boundary

            valid.append(i)

        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        """Get a sequence of frames and actions.

        Returns:
            dict with:
                - tokens: [T, S, S] int tensor of video tokens
                - actions: [T, A] float tensor of actions (if available)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length

        # Get video tokens
        tokens = self.video_tokens[start_idx:end_idx].copy()
        tokens = torch.from_numpy(tokens).long()

        result = {"tokens": tokens}

        # Get actions if available
        if self.actions:
            action_list = []
            for name in sorted(self.actions.keys()):
                action_list.append(self.actions[name][start_idx:end_idx])

            actions = np.concatenate(action_list, axis=-1)
            result["actions"] = torch.from_numpy(actions.copy()).float()

        return result


# =============================================================================
# STREAMING/ITERABLE DATASET (For very large data or on-the-fly download)
# =============================================================================

class Streaming1XDataset(IterableDataset):
    """Streaming dataset that loads chunks on-demand.

    Ideal for:
    - Very large datasets that don't fit in memory
    - Distributed training across multiple workers
    - On-the-fly data augmentation
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 16,
        chunk_size: int = 10000,  # Frames per chunk
        shuffle_chunks: bool = True,
        include_actions: bool = True,
    ):
        """
        Args:
            data_dir: Path to 1X data
            seq_length: Frames per sequence
            chunk_size: Number of frames to load at once
            shuffle_chunks: Whether to shuffle chunk order
            include_actions: Whether to include action data
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.include_actions = include_actions

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._infer_metadata()

        self.num_frames = self.metadata["num_images"]
        self.token_size = self.metadata.get("s", 16)
        self.token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))

        # Calculate chunks
        self.num_chunks = (self.num_frames - self.seq_length) // self.chunk_size + 1

        print(f"Streaming 1X dataset: {self.num_frames:,} frames in {self.num_chunks} chunks")

    def _infer_metadata(self) -> dict:
        video_path = self.data_dir / "video.bin"
        file_size = video_path.stat().st_size
        token_size = 16
        bytes_per_frame = token_size * token_size * 4
        num_frames = file_size // bytes_per_frame
        return {"num_images": num_frames, "s": token_size, "token_dtype": "uint32"}

    def _load_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a chunk of data from disk."""
        start_frame = chunk_idx * self.chunk_size
        end_frame = min(start_frame + self.chunk_size + self.seq_length, self.num_frames)

        # Memory-map and read chunk
        video_path = self.data_dir / "video.bin"
        video_mmap = np.memmap(
            video_path,
            dtype=self.token_dtype,
            mode="r",
            shape=(self.num_frames, self.token_size, self.token_size)
        )
        tokens = video_mmap[start_frame:end_frame].copy()

        # Load segment IDs for this chunk
        segment_ids = None
        segment_path = self.data_dir / "segment_ids.bin"
        if segment_path.exists():
            seg_mmap = np.memmap(segment_path, dtype=np.int32, mode="r", shape=(self.num_frames,))
            segment_ids = seg_mmap[start_frame:end_frame].copy()

        # Load actions for this chunk
        actions = None
        if self.include_actions:
            actions_dir = self.data_dir / "actions"
            if actions_dir.exists():
                action_list = []
                for name in ["joint_pos", "neck_desired", "driving_command"]:
                    path = actions_dir / f"{name}.bin"
                    if path.exists():
                        shape_map = {"joint_pos": 21, "neck_desired": 2, "driving_command": 2}
                        dim = shape_map[name]
                        mmap = np.memmap(path, dtype=np.float32, mode="r",
                                        shape=(self.num_frames, dim))
                        action_list.append(mmap[start_frame:end_frame].copy())

                if action_list:
                    actions = np.concatenate(action_list, axis=-1)

        return tokens, segment_ids, actions

    def _generate_sequences(self, tokens, segment_ids, actions) -> Iterator[dict]:
        """Generate sequences from a chunk."""
        chunk_len = len(tokens)

        for i in range(chunk_len - self.seq_length):
            # Check segment boundary
            if segment_ids is not None:
                if segment_ids[i] != segment_ids[i + self.seq_length - 1]:
                    continue

            seq_tokens = torch.from_numpy(tokens[i:i + self.seq_length].copy()).long()

            result = {"tokens": seq_tokens}

            if actions is not None:
                seq_actions = torch.from_numpy(actions[i:i + self.seq_length].copy()).float()
                result["actions"] = seq_actions

            yield result

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all sequences in the dataset."""
        # Get worker info for distributed data loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker - process all chunks
            chunk_indices = list(range(self.num_chunks))
        else:
            # Multi-worker - split chunks across workers
            per_worker = self.num_chunks // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else self.num_chunks
            chunk_indices = list(range(start, end))

        # Shuffle chunks if requested
        if self.shuffle_chunks:
            np.random.shuffle(chunk_indices)

        # Stream through chunks
        for chunk_idx in chunk_indices:
            tokens, segment_ids, actions = self._load_chunk(chunk_idx)
            yield from self._generate_sequences(tokens, segment_ids, actions)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_1x_dataloader(
    data_dir: str,
    batch_size: int = 8,
    seq_length: int = 16,
    num_workers: int = 4,
    streaming: bool = False,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create DataLoader for 1X dataset.

    Args:
        data_dir: Path to 1X data
        batch_size: Batch size
        seq_length: Sequence length
        num_workers: Number of data loading workers
        streaming: Use streaming (IterableDataset) mode
        shuffle: Shuffle data (only for non-streaming)

    Returns:
        DataLoader
    """
    if streaming:
        dataset = Streaming1XDataset(
            data_dir=data_dir,
            seq_length=seq_length,
            shuffle_chunks=shuffle,
            **kwargs
        )
        # IterableDataset doesn't support shuffle in DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataset = MemoryMapped1XDataset(
            data_dir=data_dir,
            seq_length=seq_length,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


class EpochTracker:
    """Track epochs for streaming datasets."""

    def __init__(self, dataset_size: int, batch_size: int):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.samples_per_epoch = dataset_size
        self.current_sample = 0
        self.current_epoch = 0

    def step(self, batch_size: int = None):
        """Called after each batch."""
        batch_size = batch_size or self.batch_size
        self.current_sample += batch_size

        if self.current_sample >= self.samples_per_epoch:
            self.current_epoch += 1
            self.current_sample = 0
            return True  # Epoch completed
        return False

    @property
    def progress(self) -> float:
        """Progress through current epoch (0-1)."""
        return self.current_sample / self.samples_per_epoch


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing 1X Streaming Data Loader...")

    # Check if data exists or download
    data_dir = Path("data/1x_worldmodel")

    if not data_dir.exists():
        print("\nData not found. To download:")
        print("  huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data/1x_worldmodel")
        print("\nOr run: python -c \"from genesis.pilot.streaming_1x import download_1x_dataset; download_1x_dataset()\"")

        # Create dummy data for testing
        print("\nCreating dummy data for testing...")
        data_dir.mkdir(parents=True, exist_ok=True)

        num_frames = 10000
        token_size = 16

        # Dummy video tokens
        video = np.random.randint(0, 2**18, (num_frames, token_size, token_size), dtype=np.uint32)
        video.tofile(data_dir / "video.bin")

        # Dummy segment IDs (10 segments)
        segment_ids = np.repeat(np.arange(10), num_frames // 10).astype(np.int32)
        segment_ids.tofile(data_dir / "segment_ids.bin")

        # Dummy actions
        actions_dir = data_dir / "actions"
        actions_dir.mkdir(exist_ok=True)
        joint_pos = np.random.randn(num_frames, 21).astype(np.float32)
        joint_pos.tofile(actions_dir / "joint_pos.bin")

        # Metadata
        metadata = {"num_images": num_frames, "s": token_size, "token_dtype": "uint32"}
        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Created dummy data: {num_frames} frames")

    # Test memory-mapped dataset
    print("\n--- Testing MemoryMapped1XDataset ---")
    dataset = MemoryMapped1XDataset(str(data_dir), seq_length=16, stride=8)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample tokens shape: {sample['tokens'].shape}")
    if 'actions' in sample:
        print(f"Sample actions shape: {sample['actions'].shape}")

    # Test DataLoader
    loader = create_1x_dataloader(str(data_dir), batch_size=4, seq_length=16)
    batch = next(iter(loader))
    print(f"Batch tokens shape: {batch['tokens'].shape}")

    # Test streaming dataset
    print("\n--- Testing Streaming1XDataset ---")
    streaming_loader = create_1x_dataloader(
        str(data_dir),
        batch_size=4,
        seq_length=16,
        streaming=True,
        chunk_size=2000
    )

    # Get a few batches
    for i, batch in enumerate(streaming_loader):
        if i >= 3:
            break
        print(f"Streaming batch {i}: tokens={batch['tokens'].shape}")

    # Test epoch tracking
    print("\n--- Testing EpochTracker ---")
    tracker = EpochTracker(dataset_size=len(dataset), batch_size=4)

    for i in range(100):
        epoch_done = tracker.step()
        if epoch_done:
            print(f"Epoch {tracker.current_epoch} completed at step {i}")

    print("\nAll tests passed!")
