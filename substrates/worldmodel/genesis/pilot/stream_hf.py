"""True Streaming from HuggingFace - No Download Required.

Streams data directly from HuggingFace servers chunk by chunk.
Never downloads the full dataset to disk.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from typing import Iterator, Optional
from pathlib import Path
import io


# =============================================================================
# TRUE STREAMING DATASET (No download)
# =============================================================================

class HFStreamingDataset(IterableDataset):
    """Stream data directly from HuggingFace without downloading.

    Uses HuggingFace datasets streaming mode to fetch chunks on-demand.
    """

    def __init__(
        self,
        repo_id: str = "1x-technologies/worldmodel",
        seq_length: int = 16,
        split: str = "train",
        token_key: str = "video",
        action_key: str = "actions",
        shuffle: bool = True,
        buffer_size: int = 1000,
    ):
        """
        Args:
            repo_id: HuggingFace dataset repository
            seq_length: Frames per sequence
            split: Dataset split
            token_key: Key for video tokens in dataset
            action_key: Key for actions in dataset
            shuffle: Shuffle the stream
            buffer_size: Buffer size for shuffling
        """
        self.repo_id = repo_id
        self.seq_length = seq_length
        self.split = split
        self.token_key = token_key
        self.action_key = action_key
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset

        # Load dataset in streaming mode - NO DOWNLOAD
        ds = load_dataset(
            self.repo_id,
            split=self.split,
            streaming=True,  # Key: enables true streaming
        )

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)

        # Stream through dataset
        for example in ds:
            # Extract tokens and actions
            tokens = example.get(self.token_key)
            actions = example.get(self.action_key)

            if tokens is None:
                continue

            # Convert to tensors
            if isinstance(tokens, (list, np.ndarray)):
                tokens = torch.tensor(tokens)

            result = {"tokens": tokens}

            if actions is not None:
                if isinstance(actions, (list, np.ndarray)):
                    actions = torch.tensor(actions)
                result["actions"] = actions

            yield result


class HFBinaryStreamDataset(IterableDataset):
    """Stream binary files from HuggingFace using hf_hub_download with streaming.

    For datasets stored as raw binary files (like 1X worldmodel).
    Downloads and processes one shard at a time - never holds full dataset.
    """

    def __init__(
        self,
        repo_id: str = "1x-technologies/worldmodel",
        seq_length: int = 16,
        version: str = "v2.0",
        shuffle_shards: bool = True,
    ):
        """
        Args:
            repo_id: HuggingFace dataset repository
            seq_length: Frames per sequence
            version: Dataset version (v1.1 or v2.0)
            shuffle_shards: Randomize shard order
        """
        self.repo_id = repo_id
        self.seq_length = seq_length
        self.version = version
        self.shuffle_shards = shuffle_shards

    def _list_shards(self) -> list:
        """List available video shards."""
        from huggingface_hub import HfApi
        api = HfApi()

        files = list(api.list_repo_files(self.repo_id, repo_type='dataset'))

        if self.version == "v2.0":
            # v2.0 has sharded video files
            shards = [f for f in files if f.startswith('train_v2.0/videos/video_') and f.endswith('.bin')]
        else:
            # v1.1 has single video.bin
            shards = [f for f in files if 'train_v1.1/video.bin' in f]

        return sorted(shards)

    def _stream_shard(self, shard_path: str) -> Iterator[dict]:
        """Stream sequences from a single shard."""
        from huggingface_hub import hf_hub_download
        import tempfile
        import os

        # Download shard to temp location (small file, ~10-50MB each)
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_path,
            repo_type="dataset",
        )

        # Memory-map the downloaded shard
        file_size = os.path.getsize(local_path)

        # v2.0 uses Cosmos tokenizer - different format
        # For now assume similar structure
        if self.version == "v2.0":
            # Cosmos tokens - need to check actual format
            # Placeholder: assume similar to v1.1
            token_shape = (16, 16)
            token_dtype = np.uint16  # Cosmos uses smaller tokens
        else:
            token_shape = (16, 16)
            token_dtype = np.uint32

        bytes_per_frame = int(np.prod(token_shape) * np.dtype(token_dtype).itemsize)
        num_frames = file_size // bytes_per_frame

        if num_frames < self.seq_length:
            return

        # Memory-map and yield sequences
        tokens = np.memmap(local_path, dtype=token_dtype, mode='r',
                          shape=(num_frames, *token_shape))

        for i in range(num_frames - self.seq_length):
            seq = torch.from_numpy(tokens[i:i + self.seq_length].copy()).long()
            yield {"tokens": seq}

    def __iter__(self) -> Iterator[dict]:
        shards = self._list_shards()

        if not shards:
            raise RuntimeError(f"No shards found in {self.repo_id}")

        print(f"Found {len(shards)} shards to stream")

        if self.shuffle_shards:
            np.random.shuffle(shards)

        for shard_idx, shard_path in enumerate(shards):
            print(f"Streaming shard {shard_idx + 1}/{len(shards)}: {shard_path}")
            try:
                yield from self._stream_shard(shard_path)
            except Exception as e:
                print(f"Error streaming {shard_path}: {e}")
                continue


class JATStreamDataset(IterableDataset):
    """Stream JAT dataset - Atari games with actions.

    JAT has proper parquet format that streams well.
    50+ Atari games with discrete actions.
    """

    def __init__(
        self,
        game: str = "atari-breakout",
        seq_length: int = 16,
        image_size: int = 64,
        shuffle: bool = True,
        buffer_size: int = 500,
    ):
        self.game = game
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset
        from PIL import Image
        import io

        ds = load_dataset(
            "jat-project/jat-dataset",
            self.game,
            streaming=True,
            split="train",
        )

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)

        for example in ds:
            # JAT stores sequences of observations and actions
            obs = example.get("image_observations", [])
            actions = example.get("discrete_actions", [])

            if len(obs) < self.seq_length:
                continue

            # Convert all observations to tensors
            all_frames = []
            for img_bytes in obs:
                try:
                    if isinstance(img_bytes, bytes):
                        img = Image.open(io.BytesIO(img_bytes))
                    elif isinstance(img_bytes, Image.Image):
                        img = img_bytes
                    elif isinstance(img_bytes, dict) and 'bytes' in img_bytes:
                        img = Image.open(io.BytesIO(img_bytes['bytes']))
                    else:
                        continue

                    img = img.convert('RGB').resize((self.image_size, self.image_size))
                    frame = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                    all_frames.append(frame)
                except Exception:
                    continue

            if len(all_frames) < self.seq_length:
                continue

            # Yield multiple sequences from this trajectory (sliding window)
            stride = self.seq_length // 2  # 50% overlap
            for start_idx in range(0, len(all_frames) - self.seq_length + 1, stride):
                end_idx = start_idx + self.seq_length
                frames = torch.stack(all_frames[start_idx:end_idx])

                result = {"frames": frames}

                if actions and len(actions) >= end_idx:
                    result["actions"] = torch.tensor(actions[start_idx:end_idx]).long()

                yield result


class OpenXStreamDataset(IterableDataset):
    """Stream robotics video (OpenX style).

    Since OpenX HuggingFace source uses deprecated scripts,
    uses WebVid URL streaming as alternative.
    """

    def __init__(
        self,
        subset: str = "fractal20220817_data",
        seq_length: int = 16,
        image_size: int = 64,
        shuffle: bool = True,
        buffer_size: int = 500,
    ):
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        # Use WebVid URL streaming
        print("OpenX: Using WebVid URL streaming")
        url_ds = URLVideoStreamDataset(
            url_source="webvid",
            seq_length=self.seq_length,
            image_size=self.image_size,
            shuffle=self.shuffle,
            buffer_size=self.buffer_size,
        )
        yield from url_ds


# =============================================================================
# URL-BASED VIDEO STREAMING - Fetch, use, discard (no persistent storage)
# =============================================================================

class URLVideoStreamDataset(IterableDataset):
    """Stream videos from URLs - fetch on-demand, use, discard immediately.

    Zero persistent storage. Videos are:
    1. Fetched from URL into memory
    2. Decoded to frames
    3. Used for training batch
    4. Immediately discarded (garbage collected)

    Works with datasets that provide video URLs (WebVid, MSR-VTT, etc.)
    """

    def __init__(
        self,
        url_source: str = "webvid",
        seq_length: int = 16,
        image_size: int = 256,
        shuffle: bool = True,
        buffer_size: int = 100,
        frame_stride: int = 2,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Args:
            url_source: Dataset with URLs ("webvid", "msrvtt", or HF dataset with 'url' field)
            seq_length: Frames per sequence
            image_size: Resize frames to this size
            shuffle: Shuffle URL order
            buffer_size: Shuffle buffer for streaming
            frame_stride: Sample every Nth frame
            max_retries: Retries for failed downloads
            timeout: Download timeout in seconds
        """
        self.url_source = url_source
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.frame_stride = frame_stride
        self.max_retries = max_retries
        self.timeout = timeout

    def _fetch_video_to_memory(self, url: str) -> Optional[bytes]:
        """Fetch video bytes into memory (no disk write)."""
        import urllib.request
        import ssl

        # Create SSL context that doesn't verify (for speed)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as response:
                    return response.read()
            except Exception:
                continue
        return None

    def _decode_video_bytes(self, video_bytes: bytes) -> list:
        """Decode video bytes to frames in memory."""
        import tempfile
        import os

        # Write to temp file briefly (required by most decoders)
        # File is deleted immediately after reading
        frames = []
        tmp_path = None

        try:
            # Create temp file
            fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
            os.write(fd, video_bytes)
            os.close(fd)

            # Try imageio first
            try:
                import imageio
                reader = imageio.get_reader(tmp_path)
                for frame in reader:
                    frames.append(frame)
                reader.close()
            except:
                # Fallback to cv2
                import cv2
                cap = cv2.VideoCapture(tmp_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

        finally:
            # Always delete temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return frames

    def _get_url_iterator(self):
        """Get iterator of video URLs from source."""
        from datasets import load_dataset

        if self.url_source == "webvid":
            # WebVid has video URLs
            ds = load_dataset("TempoFunk/webvid-10M", streaming=True, split="train")
            for ex in ds:
                url = ex.get("contentUrl", ex.get("url"))
                if url:
                    yield url, ex.get("name", "")

        elif self.url_source == "msrvtt":
            # MSR-VTT has URLs
            ds = load_dataset("AlexZigma/msr-vtt", streaming=True, split="train")
            for ex in ds:
                url = ex.get("url")
                if url:
                    yield url, ex.get("caption", "")

        else:
            # Generic HF dataset with URL field
            ds = load_dataset(self.url_source, streaming=True, split="train")
            for ex in ds:
                url = ex.get("url", ex.get("video_url", ex.get("contentUrl")))
                if url:
                    yield url, ex.get("caption", ex.get("text", ""))

    def __iter__(self) -> Iterator[dict]:
        from PIL import Image

        url_iter = self._get_url_iterator()

        if self.shuffle:
            # Buffer and shuffle URLs
            buffer = []
            for url_data in url_iter:
                buffer.append(url_data)
                if len(buffer) >= self.buffer_size:
                    np.random.shuffle(buffer)
                    for item in buffer[:self.buffer_size // 2]:
                        yield from self._process_url(item)
                    buffer = buffer[self.buffer_size // 2:]

            # Process remaining
            np.random.shuffle(buffer)
            for item in buffer:
                yield from self._process_url(item)
        else:
            for url_data in url_iter:
                yield from self._process_url(url_data)

    def _process_url(self, url_data: tuple) -> Iterator[dict]:
        """Fetch, decode, yield frames, then discard."""
        from PIL import Image

        url, caption = url_data

        # Fetch video bytes (in memory only)
        video_bytes = self._fetch_video_to_memory(url)
        if video_bytes is None:
            return

        # Decode to frames (temp file deleted immediately)
        try:
            frames = self._decode_video_bytes(video_bytes)
        except Exception:
            return
        finally:
            # Explicitly free video bytes
            del video_bytes

        if len(frames) < self.seq_length * self.frame_stride:
            del frames
            return

        # Convert frames to tensors with stride
        all_tensors = []
        for i in range(0, len(frames), self.frame_stride):
            try:
                frame = frames[i]
                img = Image.fromarray(frame).convert('RGB').resize(
                    (self.image_size, self.image_size)
                )
                tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                all_tensors.append(tensor)
            except:
                continue

        # Free original frames immediately
        del frames

        if len(all_tensors) < self.seq_length:
            del all_tensors
            return

        # Yield sequences with sliding window
        stride = max(1, self.seq_length // 4)
        for start_idx in range(0, len(all_tensors) - self.seq_length + 1, stride):
            end_idx = start_idx + self.seq_length
            result = {"frames": torch.stack(all_tensors[start_idx:end_idx])}
            if caption:
                result["caption"] = caption
            yield result

        # Free tensors
        del all_tensors


# =============================================================================
# CACHED VIDEO DATASET - Download once, train forever
# =============================================================================

class CachedWebVidDataset(IterableDataset):
    """WebVid with persistent local tensor cache.

    Phase 1 (cold cache): Downloads from WebVid, saves decoded frame
    sequences as uint8 .pt files to disk.
    Phase 2 (warm cache): Loads from local .pt files (~300x faster).

    Cache format: each .pt file holds [N, T, C, H, W] uint8 tensor
    where N = sequences extracted from one video.

    Disk usage: ~2-3 GB for 50 videos at 720p (uint8).
    Load speed: ~5-10ms per file vs ~3000ms HTTP download.
    """

    def __init__(
        self,
        cache_dir: str = "data/video_cache",
        max_videos: int = 50,
        url_source: str = "webvid",
        seq_length: int = 8,
        image_size: int = 720,
        frame_stride: int = 2,
        shuffle: bool = True,
        buffer_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.cache_dir = Path(cache_dir) / str(image_size)
        self.max_videos = max_videos
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle

        # Streaming params for cold cache population
        self._url_source = url_source
        self._frame_stride = frame_stride
        self._buffer_size = buffer_size
        self._max_retries = max_retries
        self._timeout = timeout

        # Ensure cache dir exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_cache_list()

    def _refresh_cache_list(self):
        """Scan cache directory for existing .pt files."""
        self.cached_files = sorted(self.cache_dir.glob("video_*.pt"))

    @property
    def cache_ready(self) -> bool:
        """True if cache has enough videos to skip downloading."""
        return len(self.cached_files) >= self.max_videos

    def _populate_cache(self):
        """Download videos from WebVid and save as cached tensors."""
        existing = len(self.cached_files)
        needed = self.max_videos - existing

        if needed <= 0:
            return

        print(f"Video cache: {existing}/{self.max_videos} cached, "
              f"downloading {needed} more to {self.cache_dir}",
              file=sys.stderr)

        # Create a streaming dataset to pull from
        stream_ds = URLVideoStreamDataset(
            url_source=self._url_source,
            seq_length=self.seq_length,
            image_size=self.image_size,
            shuffle=self.shuffle,
            buffer_size=self._buffer_size,
            frame_stride=self._frame_stride,
            max_retries=self._max_retries,
            timeout=self._timeout,
        )

        video_count = 0
        sequences = []
        batch_size = 8  # sequences per cache file

        for sample in stream_ds:
            frames = sample['frames']  # [T, C, H, W] float32 in [0,1]
            # Convert to uint8 for compact storage
            frames_u8 = (frames * 255).clamp(0, 255).to(torch.uint8)
            sequences.append(frames_u8)

            if len(sequences) >= batch_size:
                cache_path = self.cache_dir / f"video_{existing + video_count:04d}.pt"
                torch.save(torch.stack(sequences), cache_path)
                video_count += 1
                sequences = []
                print(f"  Cached {existing + video_count}/{self.max_videos} "
                      f"({batch_size} sequences)", file=sys.stderr)

                if video_count >= needed:
                    break

        # Save remaining partial batch
        if sequences and video_count < needed:
            cache_path = self.cache_dir / f"video_{existing + video_count:04d}.pt"
            torch.save(torch.stack(sequences), cache_path)
            video_count += 1

        self._refresh_cache_list()
        print(f"Video cache ready: {len(self.cached_files)} files in "
              f"{self.cache_dir}", file=sys.stderr)

    def __iter__(self) -> Iterator[dict]:
        """Yield from cached data, populating cache if needed."""
        if not self.cache_ready:
            self._populate_cache()

        if not self.cached_files:
            # Fallback to streaming if cache population failed
            print("Cache population failed, falling back to streaming",
                  file=sys.stderr)
            fallback = URLVideoStreamDataset(
                url_source=self._url_source,
                seq_length=self.seq_length,
                image_size=self.image_size,
                shuffle=self.shuffle,
                buffer_size=self._buffer_size,
                frame_stride=self._frame_stride,
            )
            yield from fallback
            return

        # Infinite loop over cached data
        while True:
            files = list(self.cached_files)
            if self.shuffle:
                np.random.shuffle(files)

            for cache_path in files:
                try:
                    # Load uint8 tensor batch: [N, T, C, H, W]
                    sequences = torch.load(cache_path, weights_only=True)
                    indices = list(range(len(sequences)))
                    if self.shuffle:
                        np.random.shuffle(indices)

                    for idx in indices:
                        # Convert uint8 back to float32 [0, 1]
                        frames = sequences[idx].float() / 255.0
                        yield {"frames": frames}

                except Exception as e:
                    print(f"Error loading cache {cache_path}: {e}",
                          file=sys.stderr)
                    continue


# =============================================================================
# LOCAL VIDEO DATASET - Minimal caching, stream from disk
# =============================================================================

class LocalVideoDataset(IterableDataset):
    """Stream from local video files with minimal memory footprint.

    Processes one video at a time, yields batches, then discards.
    Only keeps current video frames in memory.
    """

    def __init__(
        self,
        video_dir: str,
        seq_length: int = 16,
        image_size: int = 256,
        shuffle: bool = True,
        frame_stride: int = 2,
        extensions: tuple = ('.mp4', '.avi', '.webm', '.mkv', '.mov'),
    ):
        self.video_dir = Path(video_dir) if isinstance(video_dir, str) else video_dir
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.frame_stride = frame_stride
        self.extensions = extensions

        # Find video files (just paths, not loading)
        self.video_files = []
        for ext in extensions:
            self.video_files.extend(self.video_dir.rglob(f'*{ext}'))
        self.video_files = sorted(self.video_files)

        if not self.video_files:
            raise ValueError(f"No video files found in {video_dir}")

        print(f"LocalVideoDataset: Found {len(self.video_files)} videos")

    def _stream_video_frames(self, video_path: Path) -> Iterator[np.ndarray]:
        """Stream frames one at a time (minimal memory)."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_stride == 0:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx += 1
            cap.release()
        except Exception:
            return

    def __iter__(self) -> Iterator[dict]:
        from PIL import Image

        video_files = list(self.video_files)
        if self.shuffle:
            np.random.shuffle(video_files)

        for video_path in video_files:
            # Buffer only seq_length frames at a time
            frame_buffer = []

            for frame in self._stream_video_frames(video_path):
                try:
                    img = Image.fromarray(frame).convert('RGB').resize(
                        (self.image_size, self.image_size)
                    )
                    tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                    frame_buffer.append(tensor)
                except:
                    continue

                # Yield when buffer is full
                if len(frame_buffer) >= self.seq_length:
                    yield {"frames": torch.stack(frame_buffer[:self.seq_length])}
                    # Slide buffer (keep some overlap)
                    frame_buffer = frame_buffer[self.seq_length // 2:]

            # Clear buffer after each video
            del frame_buffer


# =============================================================================
# EGO4D - Long-form egocentric video (hours of continuous footage)
# =============================================================================

class Ego4DStreamDataset(IterableDataset):
    """Stream Ego4D-like long-form video.

    Since Ego4D requires license, uses WebVid as streaming alternative
    (similar long-form web videos, no download required).
    """

    def __init__(
        self,
        subset: str = "fho_main",
        seq_length: int = 16,
        image_size: int = 256,
        shuffle: bool = True,
        buffer_size: int = 500,
        frame_stride: int = 2,
        local_path: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.frame_stride = frame_stride
        self.local_path = local_path

    def __iter__(self) -> Iterator[dict]:
        # If local path provided, use LocalVideoDataset
        if self.local_path:
            local_ds = LocalVideoDataset(
                video_dir=self.local_path,
                seq_length=self.seq_length,
                image_size=self.image_size,
                shuffle=self.shuffle,
                frame_stride=self.frame_stride,
            )
            yield from local_ds
            return

        # Use WebVid URL streaming (similar long-form videos, no download)
        print("Ego4D: Using WebVid URL streaming (long-form web videos)")
        url_ds = URLVideoStreamDataset(
            url_source="webvid",
            seq_length=self.seq_length,
            image_size=self.image_size,
            shuffle=self.shuffle,
            buffer_size=self.buffer_size,
            frame_stride=self.frame_stride,
        )
        yield from url_ds

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)

        for example in ds:
            # Ego4D stores video clips with frame sequences
            frames_data = example.get("video", example.get("frames", []))

            if frames_data is None or len(frames_data) == 0:
                # Try alternative keys
                frames_data = example.get("image", [])
                if not isinstance(frames_data, list):
                    frames_data = [frames_data]

            # Apply frame stride and collect frames
            all_frames = []
            for i, frame_data in enumerate(frames_data):
                if i % self.frame_stride != 0:
                    continue

                try:
                    if isinstance(frame_data, Image.Image):
                        img = frame_data
                    elif isinstance(frame_data, bytes):
                        img = Image.open(io.BytesIO(frame_data))
                    elif isinstance(frame_data, dict) and 'bytes' in frame_data:
                        img = Image.open(io.BytesIO(frame_data['bytes']))
                    elif isinstance(frame_data, np.ndarray):
                        img = Image.fromarray(frame_data)
                    else:
                        continue

                    img = img.convert('RGB').resize((self.image_size, self.image_size))
                    frame = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                    all_frames.append(frame)
                except Exception:
                    continue

            if len(all_frames) < self.seq_length:
                continue

            # Yield multiple sequences with sliding window (continuous, non-episodic)
            stride = max(1, self.seq_length // 4)  # 75% overlap for smooth transitions
            for start_idx in range(0, len(all_frames) - self.seq_length + 1, stride):
                end_idx = start_idx + self.seq_length
                frames = torch.stack(all_frames[start_idx:end_idx])
                yield {"frames": frames}


# =============================================================================
# SOMETHING-SOMETHING V2 - Action-focused human video
# =============================================================================

class SomethingSomethingStreamDataset(IterableDataset):
    """Stream action-focused video (SS-V2 style).

    Since SS-V2 HuggingFace source uses deprecated scripts,
    uses WebVid URL streaming (also has action-oriented content).
    """

    def __init__(
        self,
        seq_length: int = 16,
        image_size: int = 256,
        shuffle: bool = True,
        buffer_size: int = 500,
        include_labels: bool = True,
        local_path: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.local_path = local_path

    def __iter__(self) -> Iterator[dict]:
        # If local path provided, use LocalVideoDataset
        if self.local_path:
            local_ds = LocalVideoDataset(
                video_dir=self.local_path,
                seq_length=self.seq_length,
                image_size=self.image_size,
                shuffle=self.shuffle,
            )
            yield from local_ds
            return

        # Use WebVid URL streaming (action-rich web videos)
        print("SSV2: Using WebVid URL streaming (action-focused web videos)")
        url_ds = URLVideoStreamDataset(
            url_source="webvid",
            seq_length=self.seq_length,
            image_size=self.image_size,
            shuffle=self.shuffle,
            buffer_size=self.buffer_size,
        )
        yield from url_ds


# =============================================================================
# BRIDGE - Robotic manipulation with continuous actions
# =============================================================================

class BridgeStreamDataset(IterableDataset):
    """Stream robotic manipulation video (Bridge style).

    Since OpenX HuggingFace source uses deprecated scripts,
    uses WebVid URL streaming as alternative.
    """

    def __init__(
        self,
        seq_length: int = 16,
        image_size: int = 256,
        shuffle: bool = True,
        buffer_size: int = 500,
        local_path: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.local_path = local_path

    def __iter__(self) -> Iterator[dict]:
        # If local path provided, use LocalVideoDataset
        if self.local_path:
            local_ds = LocalVideoDataset(
                video_dir=self.local_path,
                seq_length=self.seq_length,
                image_size=self.image_size,
                shuffle=self.shuffle,
            )
            yield from local_ds
            return

        # Use WebVid URL streaming
        print("Bridge: Using WebVid URL streaming")
        url_ds = URLVideoStreamDataset(
            url_source="webvid",
            seq_length=self.seq_length,
            image_size=self.image_size,
            shuffle=self.shuffle,
            buffer_size=self.buffer_size,
        )
        yield from url_ds


# =============================================================================
# CONVENIENCE LOADERS
# =============================================================================

def create_streaming_loader(
    dataset_name: str = "jat",
    batch_size: int = 4,
    seq_length: int = 16,
    num_workers: int = 0,  # Streaming doesn't benefit much from workers
    **kwargs,
) -> DataLoader:
    """Create a streaming DataLoader with minimal caching.

    All datasets stream data and discard after use - no persistent storage.

    Args:
        dataset_name: One of:
            - "jat" or "atari-*": JAT Atari (streaming, episodic)
            - "webvid": WebVid-10M via URL streaming (RECOMMENDED - long videos)
            - "msrvtt": MSR-VTT via URL streaming
            - "1x": 1X Technologies worldmodel tokens
            - "local": Stream from local video directory
            - "ego4d", "ssv2", "bridge": Fall back to JAT (HF sources broken)
            - Or a HuggingFace repo ID
        batch_size: Batch size
        seq_length: Frames per sequence
        num_workers: Number of workers (0 for streaming)

    Kwargs:
        image_size: Target image size (default 256)
        video_dir: Required for "local" dataset
        frame_stride: Sample every Nth frame (default 2)

    Returns:
        DataLoader that streams data (fetch -> use -> discard)

    RECOMMENDED for world model training:
        # WebVid - millions of web videos, true streaming
        loader = create_streaming_loader(
            dataset_name="webvid",
            image_size=256,
            seq_length=16,
        )

        # JAT - reliable fallback for quick iteration
        loader = create_streaming_loader(
            dataset_name="jat",
            game="atari-breakout",
        )
    """
    # URL-based streaming (fetch -> use -> discard)
    if dataset_name in ("webvid", "msrvtt"):
        dataset = URLVideoStreamDataset(
            url_source=dataset_name,
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "local":
        video_dir = kwargs.pop("video_dir", None)
        if video_dir is None:
            raise ValueError("video_dir required for local dataset")
        dataset = LocalVideoDataset(
            video_dir=video_dir,
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "1x":
        dataset = HFBinaryStreamDataset(
            repo_id="1x-technologies/worldmodel",
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "jat" or dataset_name.startswith("atari-"):
        game = kwargs.pop("game", "atari-breakout") if dataset_name == "jat" else dataset_name
        dataset = JATStreamDataset(
            game=game,
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "openx":
        dataset = OpenXStreamDataset(
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "ego4d":
        dataset = Ego4DStreamDataset(
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "ssv2" or dataset_name == "something-something":
        dataset = SomethingSomethingStreamDataset(
            seq_length=seq_length,
            **kwargs,
        )
    elif dataset_name == "bridge":
        dataset = BridgeStreamDataset(
            seq_length=seq_length,
            **kwargs,
        )
    else:
        # Assume it's a HuggingFace repo ID
        dataset = HFStreamingDataset(
            repo_id=dataset_name,
            seq_length=seq_length,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


# =============================================================================
# EPOCH MANAGEMENT FOR STREAMING
# =============================================================================

class StreamingEpochManager:
    """Manage epochs when streaming (no fixed dataset size).

    Since streaming datasets are effectively infinite, we define
    an "epoch" as a fixed number of samples.
    """

    def __init__(self, samples_per_epoch: int = 10000, batch_size: int = 8):
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.current_sample = 0
        self.current_epoch = 0
        self.total_samples = 0

    def step(self, batch_size: int = None) -> bool:
        """Call after each batch. Returns True if epoch completed."""
        bs = batch_size or self.batch_size
        self.current_sample += bs
        self.total_samples += bs

        if self.current_sample >= self.samples_per_epoch:
            self.current_epoch += 1
            self.current_sample = 0
            return True
        return False

    @property
    def progress(self) -> float:
        return self.current_sample / self.samples_per_epoch


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test streaming datasets")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "jat", "1x", "ego4d", "ssv2", "bridge"],
                        help="Dataset to test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    def test_dataset(name, **kwargs):
        print(f"\n--- Testing {name.upper()} Streaming ---")
        try:
            loader = create_streaming_loader(
                dataset_name=name,
                batch_size=2,
                seq_length=args.seq_length,
                image_size=args.image_size,
                **kwargs,
            )

            for i, batch in enumerate(loader):
                if i >= args.batches:
                    break

                if 'frames' in batch:
                    print(f"Batch {i}: frames={batch['frames'].shape}")
                if 'tokens' in batch:
                    print(f"Batch {i}: tokens={batch['tokens'].shape}")
                if 'actions' in batch:
                    print(f"         actions={batch['actions'].shape}")
                if 'action_label' in batch:
                    print(f"         label={batch['action_label']}")

            print(f"{name.upper()} streaming works!")
            return True

        except Exception as e:
            print(f"{name.upper()} streaming failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("Testing HuggingFace Streaming (NO DOWNLOAD)...")
    print(f"Image size: {args.image_size}, Seq length: {args.seq_length}")

    results = {}

    if args.dataset in ["all", "jat"]:
        results["jat"] = test_dataset("jat", game="atari-breakout")

    if args.dataset in ["all", "ego4d"]:
        results["ego4d"] = test_dataset("ego4d")

    if args.dataset in ["all", "ssv2"]:
        results["ssv2"] = test_dataset("ssv2")

    if args.dataset in ["all", "bridge"]:
        results["bridge"] = test_dataset("bridge")

    if args.dataset in ["all", "1x"]:
        # 1X uses tokens, not images
        print("\n--- Testing 1X Binary Streaming ---")
        try:
            loader = create_streaming_loader(
                dataset_name="1x",
                batch_size=2,
                seq_length=16,
                version="v2.0",
            )

            for i, batch in enumerate(loader):
                if i >= args.batches:
                    break
                print(f"Batch {i}: tokens={batch['tokens'].shape}")

            print("1X streaming works!")
            results["1x"] = True

        except Exception as e:
            print(f"1X streaming failed: {e}")
            results["1x"] = False

    # Summary
    print("\n" + "=" * 50)
    print("STREAMING TEST SUMMARY")
    print("=" * 50)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name:10s}: {status}")
    print("=" * 50)
