"""Object Permanence Metrics: Beyond MSE.

Genesis core thesis: Object permanence without massive parameters.
This module provides metrics specifically for object permanence:

1. Object Detection: Find objects in frames
2. Object Tracking: Track objects across time
3. Occlusion Detection: Detect when objects are occluded
4. Reappearance Accuracy: Measure prediction quality after occlusion
5. Identity Preservation: Do objects maintain their identity?
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import ndimage


@dataclass
class ObjectState:
    """State of a detected object."""
    id: int
    position: Tuple[float, float]  # (y, x) center
    size: float  # approximate size
    color: Tuple[float, float, float]  # RGB
    visible: bool
    frame_idx: int


@dataclass
class PermanenceMetrics:
    """Object permanence evaluation metrics."""
    # Basic
    mse: float
    psnr: float

    # Object permanence specific
    object_count_accuracy: float  # % of frames with correct object count
    position_error: float  # Average position error for tracked objects
    reappearance_accuracy: float  # Accuracy on frames after occlusion
    identity_preservation: float  # % of objects maintaining identity

    # Occlusion specific
    occlusion_frames: int  # Number of frames with occlusions
    recovery_mse: float  # MSE on recovery frames (after occlusion ends)


def detect_objects_simple(frame: torch.Tensor, threshold: float = 0.3) -> List[dict]:
    """Simple object detection using thresholding and connected components.

    Args:
        frame: [C, H, W] or [H, W] tensor
        threshold: Intensity threshold for object detection

    Returns:
        List of objects with position, size, color
    """
    if frame.dim() == 3:
        # RGB: convert to grayscale for detection
        gray = frame.mean(dim=0)
        rgb = frame
    else:
        gray = frame
        rgb = frame.unsqueeze(0).expand(3, -1, -1)

    # Threshold
    binary = (gray > threshold).cpu().numpy().astype(np.int32)

    # Connected components
    labeled, num_objects = ndimage.label(binary)

    objects = []
    for obj_id in range(1, num_objects + 1):
        mask = labeled == obj_id
        if mask.sum() < 4:  # Ignore tiny noise
            continue

        # Get position (center of mass)
        y_coords, x_coords = np.where(mask)
        y_center = y_coords.mean()
        x_center = x_coords.mean()

        # Get size
        size = mask.sum()

        # Get color (average in mask)
        mask_tensor = torch.from_numpy(mask).to(frame.device)
        color = (rgb * mask_tensor.unsqueeze(0)).sum(dim=(1, 2)) / (mask.sum() + 1e-6)

        objects.append({
            'id': obj_id,
            'position': (y_center, x_center),
            'size': size,
            'color': color.cpu().tolist(),
            'mask': mask,
        })

    return objects


def track_objects(objects_t0: List[dict], objects_t1: List[dict],
                  max_distance: float = 20.0) -> dict:
    """Simple object tracking by nearest neighbor matching.

    Args:
        objects_t0: Objects at time t
        objects_t1: Objects at time t+1
        max_distance: Maximum distance for matching

    Returns:
        Dictionary mapping t0 object IDs to t1 object IDs (or None if lost)
    """
    if not objects_t0 or not objects_t1:
        return {}

    # Compute distance matrix
    distances = np.zeros((len(objects_t0), len(objects_t1)))
    for i, obj0 in enumerate(objects_t0):
        for j, obj1 in enumerate(objects_t1):
            y0, x0 = obj0['position']
            y1, x1 = obj1['position']
            distances[i, j] = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

    # Greedy matching
    matching = {}
    used_t1 = set()

    for i in range(len(objects_t0)):
        best_j = None
        best_dist = max_distance

        for j in range(len(objects_t1)):
            if j not in used_t1 and distances[i, j] < best_dist:
                best_dist = distances[i, j]
                best_j = j

        if best_j is not None:
            matching[objects_t0[i]['id']] = objects_t1[best_j]['id']
            used_t1.add(best_j)
        else:
            matching[objects_t0[i]['id']] = None  # Object lost/occluded

    return matching


def detect_occlusion_events(frames: torch.Tensor, threshold: float = 0.3) -> List[dict]:
    """Detect occlusion events in a sequence.

    An occlusion event is when an object disappears and reappears.

    Args:
        frames: [T, C, H, W] tensor

    Returns:
        List of occlusion events with start/end frames
    """
    T = frames.shape[0]

    # Track objects across all frames
    all_objects = []
    for t in range(T):
        objects = detect_objects_simple(frames[t], threshold)
        all_objects.append(objects)

    # Find object tracks
    tracks = {}  # object_id -> list of (frame_idx, position)

    # Initialize with first frame
    for obj in all_objects[0]:
        tracks[obj['id']] = [(0, obj['position'], True)]

    # Track through sequence
    current_ids = {obj['id']: obj['id'] for obj in all_objects[0]}

    for t in range(1, T):
        prev_objects = all_objects[t - 1]
        curr_objects = all_objects[t]

        matching = track_objects(prev_objects, curr_objects)

        # Update tracks
        new_current_ids = {}
        for prev_id, curr_id in matching.items():
            track_id = current_ids.get(prev_id, prev_id)

            if curr_id is not None:
                # Object still visible
                curr_obj = next(o for o in curr_objects if o['id'] == curr_id)
                tracks.setdefault(track_id, []).append((t, curr_obj['position'], True))
                new_current_ids[curr_id] = track_id
            else:
                # Object occluded
                tracks.setdefault(track_id, []).append((t, None, False))

        current_ids = new_current_ids

    # Find occlusion events (visible -> not visible -> visible)
    occlusion_events = []
    for track_id, track in tracks.items():
        visible_sequence = [v for _, _, v in track]

        i = 0
        while i < len(visible_sequence):
            if visible_sequence[i]:
                # Find occlusion start
                j = i + 1
                while j < len(visible_sequence) and visible_sequence[j]:
                    j += 1

                if j < len(visible_sequence):
                    # Found occlusion start
                    occlusion_start = j

                    # Find occlusion end
                    k = j + 1
                    while k < len(visible_sequence) and not visible_sequence[k]:
                        k += 1

                    if k < len(visible_sequence):
                        # Found reappearance
                        occlusion_events.append({
                            'track_id': track_id,
                            'start_frame': occlusion_start,
                            'end_frame': k,
                            'duration': k - occlusion_start,
                        })
                    i = k
                else:
                    i = j
            else:
                i += 1

    return occlusion_events


def compute_permanence_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    detection_threshold: float = 0.3,
) -> PermanenceMetrics:
    """Compute comprehensive object permanence metrics.

    Args:
        predictions: [B, T, C, H, W] or [T, C, H, W]
        targets: [B, T, C, H, W] or [T, C, H, W]
        detection_threshold: Threshold for object detection

    Returns:
        PermanenceMetrics dataclass with all metrics
    """
    if predictions.dim() == 4:
        predictions = predictions.unsqueeze(0)
        targets = targets.unsqueeze(0)

    B, T, C, H, W = predictions.shape

    # Basic metrics
    mse = F.mse_loss(predictions, targets).item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    # Per-batch analysis
    all_count_correct = 0
    all_position_errors = []
    all_recovery_mse = []
    all_identity_preserved = 0
    all_identity_total = 0
    total_occlusion_frames = 0

    for b in range(B):
        pred_seq = predictions[b]  # [T, C, H, W]
        target_seq = targets[b]

        # Detect occlusion events in ground truth
        occlusion_events = detect_occlusion_events(target_seq, detection_threshold)
        total_occlusion_frames += sum(e['duration'] for e in occlusion_events)

        # Object count accuracy
        for t in range(T):
            pred_objects = detect_objects_simple(pred_seq[t], detection_threshold)
            target_objects = detect_objects_simple(target_seq[t], detection_threshold)
            if len(pred_objects) == len(target_objects):
                all_count_correct += 1

        # Position tracking error
        prev_pred_objects = None
        prev_target_objects = None

        for t in range(T):
            pred_objects = detect_objects_simple(pred_seq[t], detection_threshold)
            target_objects = detect_objects_simple(target_seq[t], detection_threshold)

            if prev_target_objects and target_objects:
                # Match prediction to target
                target_matching = track_objects(prev_target_objects, target_objects)
                pred_matching = track_objects(prev_pred_objects or [], pred_objects)

                # Compare positions
                for tid, matched_tid in target_matching.items():
                    if matched_tid is not None:
                        target_obj = next((o for o in target_objects if o['id'] == matched_tid), None)
                        if target_obj:
                            # Find corresponding predicted object
                            pred_obj = next((o for o in pred_objects
                                           if abs(o['position'][0] - target_obj['position'][0]) < 15
                                           and abs(o['position'][1] - target_obj['position'][1]) < 15),
                                          None)
                            if pred_obj:
                                ty, tx = target_obj['position']
                                py, px = pred_obj['position']
                                error = np.sqrt((ty - py) ** 2 + (tx - px) ** 2)
                                all_position_errors.append(error)

            prev_pred_objects = pred_objects
            prev_target_objects = target_objects

        # Recovery MSE (frames right after occlusion ends)
        for event in occlusion_events:
            end_frame = event['end_frame']
            if end_frame < T:
                recovery_mse = F.mse_loss(pred_seq[end_frame], target_seq[end_frame]).item()
                all_recovery_mse.append(recovery_mse)

        # Identity preservation (color consistency)
        for event in occlusion_events:
            start = event['start_frame']
            end = event['end_frame']

            if start > 0 and end < T:
                before_objects = detect_objects_simple(target_seq[start - 1], detection_threshold)
                after_objects = detect_objects_simple(target_seq[end], detection_threshold)

                pred_before = detect_objects_simple(pred_seq[start - 1], detection_threshold)
                pred_after = detect_objects_simple(pred_seq[end], detection_threshold)

                if before_objects and after_objects and pred_after:
                    # Check if predicted objects after occlusion match before
                    for before_obj in before_objects:
                        for pred_obj in pred_after:
                            # Compare colors
                            bc = np.array(before_obj['color'])
                            pc = np.array(pred_obj['color'])
                            color_diff = np.linalg.norm(bc - pc)

                            all_identity_total += 1
                            if color_diff < 0.3:  # Similar color
                                all_identity_preserved += 1

    # Aggregate metrics
    total_frames = B * T
    object_count_accuracy = all_count_correct / total_frames

    position_error = np.mean(all_position_errors) if all_position_errors else 0.0
    recovery_mse_val = np.mean(all_recovery_mse) if all_recovery_mse else mse
    identity_preservation = all_identity_preserved / max(all_identity_total, 1)

    return PermanenceMetrics(
        mse=mse,
        psnr=psnr,
        object_count_accuracy=object_count_accuracy,
        position_error=position_error,
        reappearance_accuracy=1.0 - min(recovery_mse_val * 10, 1.0),  # Convert MSE to accuracy
        identity_preservation=identity_preservation,
        occlusion_frames=total_occlusion_frames,
        recovery_mse=recovery_mse_val,
    )


def benchmark_permanence(model, dataset, device, num_samples=100):
    """Run full permanence benchmark on a model.

    Args:
        model: World model with .generate() method
        dataset: Video dataset
        device: torch device
        num_samples: Number of sequences to evaluate

    Returns:
        PermanenceMetrics aggregated over all samples
    """
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_metrics = []

    for i, frames in enumerate(loader):
        if i >= num_samples:
            break

        frames = frames.to(device)
        B, T, C, H, W = frames.shape

        # Generate from seed
        seed = frames[:, :2]
        with torch.no_grad():
            if hasattr(model, 'generate'):
                generated = model.generate(seed, num_steps=T - 2)
            else:
                generated = model(frames)

        # Compute metrics
        targets = frames[:, 2:]
        metrics = compute_permanence_metrics(generated, targets)
        all_metrics.append(metrics)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} sequences")

    # Aggregate
    return PermanenceMetrics(
        mse=np.mean([m.mse for m in all_metrics]),
        psnr=np.mean([m.psnr for m in all_metrics]),
        object_count_accuracy=np.mean([m.object_count_accuracy for m in all_metrics]),
        position_error=np.mean([m.position_error for m in all_metrics]),
        reappearance_accuracy=np.mean([m.reappearance_accuracy for m in all_metrics]),
        identity_preservation=np.mean([m.identity_preservation for m in all_metrics]),
        occlusion_frames=sum(m.occlusion_frames for m in all_metrics),
        recovery_mse=np.mean([m.recovery_mse for m in all_metrics]),
    )


def print_permanence_report(metrics: PermanenceMetrics, model_name: str = "Model"):
    """Print formatted permanence report."""
    print("\n" + "=" * 60)
    print(f"OBJECT PERMANENCE REPORT: {model_name}")
    print("=" * 60)

    print("\nBasic Metrics:")
    print(f"  MSE:                    {metrics.mse:.6f}")
    print(f"  PSNR:                   {metrics.psnr:.2f} dB")

    print("\nObject Permanence Metrics:")
    print(f"  Object Count Accuracy:  {metrics.object_count_accuracy * 100:.1f}%")
    print(f"  Position Error:         {metrics.position_error:.2f} pixels")
    print(f"  Reappearance Accuracy:  {metrics.reappearance_accuracy * 100:.1f}%")
    print(f"  Identity Preservation:  {metrics.identity_preservation * 100:.1f}%")

    print("\nOcclusion Statistics:")
    print(f"  Occlusion Frames:       {metrics.occlusion_frames}")
    print(f"  Recovery MSE:           {metrics.recovery_mse:.6f}")

    print("=" * 60)


if __name__ == '__main__':
    print("Testing Object Permanence Metrics...")

    # Test with synthetic data
    from genesis.pilot.video_data import SyntheticVideoDataset

    dataset = SyntheticVideoDataset(num_sequences=10, seq_length=16, num_objects=2)
    frames = dataset[0].unsqueeze(0)  # [1, T, C, H, W]

    print(f"Test sequence shape: {frames.shape}")

    # Detect objects
    objects = detect_objects_simple(frames[0, 0])
    print(f"Objects in frame 0: {len(objects)}")
    for obj in objects:
        print(f"  Object {obj['id']}: pos={obj['position']}, size={obj['size']}")

    # Detect occlusions
    events = detect_occlusion_events(frames[0])
    print(f"\nOcclusion events: {len(events)}")
    for event in events:
        print(f"  Track {event['track_id']}: frames {event['start_frame']}-{event['end_frame']}")

    # Compute metrics (using same sequence as target = perfect prediction)
    metrics = compute_permanence_metrics(frames, frames)
    print_permanence_report(metrics, "Perfect Prediction")

    # Compute metrics with noise (simulating imperfect prediction)
    noisy = frames + torch.randn_like(frames) * 0.1
    noisy = noisy.clamp(0, 1)
    metrics_noisy = compute_permanence_metrics(noisy, frames)
    print_permanence_report(metrics_noisy, "Noisy Prediction")

    print("\nAll tests passed!")
