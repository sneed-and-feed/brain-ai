"""
scripts/train_diffusion_policy.py: Train 1D CNN Diffusion Policy on Clean Franka Demos.

Trains the DiffusionPolicy1D model on 180 nominal pick-and-place and reach demonstrations,
saving the model checkpoint to checkpoints/embodied/diffusion_policy_franka.pt.
"""

import os
import sys
import time
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brain_ai.models.diffusion_policy import (
    DiffusionRoboticsPolicy,
    train_diffusion_policy,
    save_diffusion_checkpoint
)


def main():
    demo_path = "data/embodied_demos/franka_nominal_demos.pt"
    ckpt_dir = "checkpoints/embodied"
    ckpt_path = os.path.join(ckpt_dir, "diffusion_policy_franka.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstration dataset not found at {demo_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"TRAINING 1D TEMPORAL CNN DIFFUSION POLICY (Device: {device})")
    print(f"Dataset: {demo_path}")
    print(f"Output Checkpoint: {ckpt_path}")
    print("=" * 70)

    dataset = torch.load(demo_path, weights_only=True)
    obs = dataset["observations"]
    actions = dataset["actions"]
    print(f"Loaded {len(obs)} transitions across {len(dataset['episode_lengths'])} episodes.")

    # Initialize policy architecture (down_dims=(32, 64, 128) for fast high-efficiency inference)
    policy = DiffusionRoboticsPolicy(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        exec_horizon=8,
        down_dims=(32, 64, 128),
        cond_dim=128,
        num_train_timesteps=100,
        num_ddim_steps=10,
        simulated_latency_ms=45.0,
        device=device
    )

    t0 = time.perf_counter()
    losses = train_diffusion_policy(
        policy=policy,
        dataset=dataset,
        epochs=15,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        verbose=True
    )
    dur = time.perf_counter() - t0

    print(f"\nTraining completed in {dur:.2f}s across 15 epochs.")
    print(f"Final training loss: {losses[-1]:.6f}")

    save_diffusion_checkpoint(policy, ckpt_path)
    print(f"Checkpoint successfully saved to {ckpt_path}")


if __name__ == "__main__":
    main()
