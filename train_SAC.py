from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent import BasicAgent
from SAC import SacAgent
from train import PoolTrainEnv, TrainStatsCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC NewAgent in PoolEnv")
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--player-role", choices=["A", "B"], default="A",
                        help="Player identity controlled by SacAgent during training")
    parser.add_argument("--ball-type", choices=["solid", "stripe"], default="solid",
                        help="Target ball type assigned to SacAgent for this training run")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum turns (shots) per episode to avoid stalls")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episode interval for logging aggregated statistics")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Episode interval for checkpointing. Set 0 to disable")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to store model checkpoints")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Optional SAC checkpoint (.zip) to resume training from")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device string, e.g. 'cuda:0'. Defaults to auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-env-noise", action="store_true",
                        help="Disable environment shot noise during training")
    parser.add_argument("--verbose", action="store_true", help="Print SAC update metrics")
    parser.add_argument("--tb-logdir", type=str, default="runs/sac",
                        help="TensorBoard log directory (set empty string to disable logging)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    player_role = args.player_role
    ball_type = args.ball_type
    if player_role == "A":
        env_target_ball = ball_type
    else:
        env_target_ball = "stripe" if ball_type == "solid" else "solid"

    role_suffix = f"player{player_role}_{ball_type}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, role_suffix)
    tb_log_dir: Optional[str] = None
    if args.tb_logdir:
        tb_log_dir = os.path.join(args.tb_logdir, role_suffix)

    opponent = BasicAgent()
    training_agent = SacAgent(device=args.device, deterministic_eval=False, training=True)

    train_env = PoolTrainEnv(
        helper_agent=training_agent,
        opponent=opponent,
        target_ball=env_target_ball,
        helper_player=player_role,
        max_steps=args.max_steps,
        disable_env_noise=args.disable_env_noise,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    if tb_log_dir is not None:
        os.makedirs(tb_log_dir, exist_ok=True)

    print(
        f"Training SacAgent as Player {player_role} with {ball_type} balls. "
        f"Checkpoints will be stored in {checkpoint_dir}."
    )
    if tb_log_dir is not None:
        print(f"TensorBoard logs will be stored in {tb_log_dir}.")

    writer: Optional[SummaryWriter] = None
    if tb_log_dir is not None:
        writer = SummaryWriter(log_dir=tb_log_dir)

    checkpoint_path = None
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        training_agent.load(checkpoint_path)
        training_agent.model.set_env(train_env)
        training_agent.model.verbose = 1 if args.verbose else 0
        if tb_log_dir is not None:
            training_agent.model.tensorboard_log = tb_log_dir
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        training_agent.build_model(
            train_env,
            tensorboard_log=tb_log_dir,
            verbose=1 if args.verbose else 0,
        )

    callback = TrainStatsCallback(
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        log_interval=args.log_interval,
        verbose=1 if args.verbose else 0,
    )

    total_timesteps = int(args.episodes * args.max_steps)
    print(f"Starting SAC training for {total_timesteps} timesteps.")
    training_agent.learn(total_timesteps=total_timesteps, callback=callback)

    if writer is not None:
        writer.close()

    train_env.close()


if __name__ == "__main__":
    main()
