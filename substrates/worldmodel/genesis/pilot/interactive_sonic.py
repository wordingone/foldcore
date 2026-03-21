"""Interactive Sonic - Play through Genesis with keyboard controls.

Use arrow keys to control Sonic. The world model predicts the next frame
based on your actions.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import pygame
from pathlib import Path

from genesis.pilot.action_model import ActionConditionedModel
from genesis.pilot.video_data import get_video_dataset


# Action mapping
# 0 = none, 1 = up, 2 = down, 3 = left, 4 = right
ACTION_NAMES = ['NONE', 'UP', 'DOWN', 'LEFT', 'RIGHT']


def load_or_create_model(checkpoint_path, device):
    """Load action-conditioned model."""
    model = ActionConditionedModel(
        base_channels=48,
        num_slots=12,
        slot_dim=64,
        slot_decay=0.95,
        num_actions=5,
        continuous_actions=False,
    )

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        # Try loading base infinite horizon model weights (compatible subset)
        base_path = Path('checkpoints/sonic/best_sonic.pt')
        if base_path.exists():
            base_ckpt = torch.load(base_path, map_location=device, weights_only=False)
            # Load compatible weights
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in base_ckpt['model_state_dict'].items()
                         if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict)
            print(f"Loaded base weights from {base_path} ({len(pretrained)} layers)")
        else:
            print("No checkpoint found - using random weights")

    model = model.to(device)
    model.eval()
    return model


def frame_to_surface(frame, scale=4):
    """Convert tensor frame to pygame surface."""
    # frame: [C, H, W] tensor
    frame = frame.cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    frame = frame.transpose(1, 2, 0)  # [H, W, C]

    # Create pygame surface
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    # Scale up
    if scale > 1:
        w, h = surface.get_size()
        surface = pygame.transform.scale(surface, (w * scale, h * scale))

    return surface


def get_action_from_keys(keys):
    """Map pygame keys to action index."""
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 1  # UP
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return 2  # DOWN
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 3  # LEFT
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 4  # RIGHT
    return 0  # NONE


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Play Sonic through Genesis")
    parser.add_argument('--checkpoint', default='checkpoints/sonic/action_sonic.pt')
    parser.add_argument('--scale', type=int, default=6, help='Display scale')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("GENESIS INTERACTIVE SONIC")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow keys or WASD - Move Sonic")
    print("  R - Reset to new seed")
    print("  ESC or Q - Quit")
    print("=" * 60)

    # Initialize
    device = args.device
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading model...")
    model = load_or_create_model(args.checkpoint, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset for seed frames
    print("\nLoading Sonic dataset...")
    dataset = get_video_dataset("tinyworlds:sonic", seq_length=16)
    print(f"Dataset: {len(dataset)} sequences")

    # Initialize pygame
    pygame.init()

    resolution = 64
    window_size = resolution * args.scale
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Genesis Plays Sonic - Use Arrow Keys")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Get initial seed
    seed_idx = np.random.randint(len(dataset))
    seed_frames = dataset[seed_idx][:2]  # [2, C, H, W]

    # Initialize state
    prev_frame = seed_frames[0:1].to(device)  # [1, C, H, W]
    curr_frame = seed_frames[1:2].to(device)  # [1, C, H, W]
    model.reset_state()

    # Warm up with seed frames
    with torch.no_grad():
        _ = model.step(prev_frame, curr_frame, None)

    running = True
    frame_count = 0
    current_action = 0

    print("\nStarting game loop...")
    print("Press arrow keys to control Sonic!")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    seed_idx = np.random.randint(len(dataset))
                    seed_frames = dataset[seed_idx][:2]
                    prev_frame = seed_frames[0:1].to(device)
                    curr_frame = seed_frames[1:2].to(device)
                    model.reset_state()
                    with torch.no_grad():
                        _ = model.step(prev_frame, curr_frame, None)
                    frame_count = 0
                    print(f"\nReset to sequence {seed_idx}")

        # Get current action from keyboard
        keys = pygame.key.get_pressed()
        current_action = get_action_from_keys(keys)

        # Generate next frame
        action_tensor = torch.tensor([current_action], device=device)

        with torch.no_grad():
            next_frame = model.step(prev_frame, curr_frame, action_tensor)

        # Update state
        prev_frame = curr_frame
        curr_frame = next_frame
        frame_count += 1

        # Render
        surface = frame_to_surface(curr_frame[0], scale=args.scale)
        screen.blit(surface, (0, 0))

        # Draw HUD
        action_text = font.render(f"Action: {ACTION_NAMES[current_action]}", True, (255, 255, 255))
        frame_text = font.render(f"Frame: {frame_count}", True, (255, 255, 255))
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (255, 255, 255))

        # Semi-transparent background for text
        hud_surface = pygame.Surface((120, 60), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 128))
        screen.blit(hud_surface, (5, 5))

        screen.blit(action_text, (10, 10))
        screen.blit(frame_text, (10, 28))
        screen.blit(fps_text, (10, 46))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print(f"\nGenerated {frame_count} frames")
    print("Thanks for playing!")


if __name__ == '__main__':
    main()
