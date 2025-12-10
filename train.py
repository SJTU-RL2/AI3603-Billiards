"""Train a PPO-based NewAgent inside the PoolEnv environment.

This script demonstrates how to run self-play style training where the PPO
agent controls Player A and faces the provided BasicAgent as Player B.
The reward shaping relies on ``analyze_shot_for_reward`` from ``agent.py``.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np

from poolenv import PoolEnv, save_balls_state
from agent import (
    BasicAgent,
    NewAgent,
    PPOConfig,
    analyze_shot_for_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO NewAgent in PoolEnv")
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--target-ball", choices=["solid", "stripe"], default="solid",
                        help="Initial target ball assignment for Player A")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum turns (shots) per episode to avoid stalls")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episode interval for logging aggregated statistics")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Episode interval for checkpointing. Set 0 to disable")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to store model checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device string, e.g. 'cuda:0'. Defaults to auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-env-noise", action="store_true",
                        help="Disable environment shot noise during training")
    parser.add_argument("--verbose", action="store_true", help="Print PPO update metrics")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = PoolEnv()
    if args.disable_env_noise:
        env.enable_noise = False

    ppo_config = PPOConfig()
    agent = NewAgent(config=ppo_config, device=args.device, training=True)
    opponent = BasicAgent()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    episode_rewards = []
    best_mean_reward: Optional[float] = None

    for episode in range(1, args.episodes + 1):
        env.reset(target_ball=args.target_ball)
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < args.max_steps:
            player = env.get_curr_player()
            balls, targets, table = env.get_observation(player)

            if player == "A":
                pre_shot_state = save_balls_state(env.balls)
                action = agent.decision(balls, targets, table)
            else:
                action = opponent.decision(balls, targets, table)

            env.take_shot(action)
            done, _ = env.get_done()

            if player == "A":
                shot = env.shot_record[-1] if len(env.shot_record) > 0 else None
                reward = 0.0
                if shot is not None:
                    reward = analyze_shot_for_reward(shot, pre_shot_state, targets)
                total_reward += reward

                next_balls = next_targets = next_table = None
                if not done:
                    next_balls, next_targets, next_table = env.get_observation("A")

                metrics = agent.observe(
                    reward=reward,
                    done=done,
                    next_balls=next_balls,
                    next_targets=next_targets,
                    table=next_table,
                )
                if metrics is not None and args.verbose:
                    print(
                        f"[Episode {episode:04d}] PPO update #{metrics['updates']}: "
                        f"policy_loss={metrics['policy_loss']:.4f}, "
                        f"value_loss={metrics['value_loss']:.4f}, "
                        f"entropy={metrics['entropy']:.4f}, "
                        f"samples={metrics['samples']}"
                    )

            steps += 1

        episode_rewards.append(total_reward)

        if episode % args.log_interval == 0 or episode == 1:
            recent_rewards = episode_rewards[-args.log_interval:]
            mean_reward = float(np.mean(recent_rewards))
            print(
                f"[Episode {episode:04d}] mean_reward={mean_reward:.2f} "
                f"(last {len(recent_rewards)})"
            )
            best_mean_reward = (
                mean_reward
                if best_mean_reward is None or mean_reward > best_mean_reward
                else best_mean_reward
            )

        if args.save_every > 0 and episode % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"ppo_newagent_ep{episode:04d}.pt"
            )
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    if best_mean_reward is not None:
        print(f"Training completed. Best mean reward: {best_mean_reward:.2f}")
    else:
        print("Training completed. No episodes logged for statistics.")


if __name__ == "__main__":
    main()
