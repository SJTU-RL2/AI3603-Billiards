"""Train a PPO-based NewAgent inside the PoolEnv environment.

This script demonstrates how to run self-play style training where the PPO
agent controls Player A and faces the provided BasicAgent as Player B.
The reward shaping relies on ``analyze_shot_for_reward`` from ``agent.py``.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym

from poolenv import PoolEnv, save_balls_state
from agent import (
    BasicAgent,
    NewAgent,
    analyze_shot_for_reward,
)
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class PoolTrainEnv(gym.Env):
    """Gymnasium 环境包装器，使 PoolEnv 可用于 SB3 PPO 训练。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        helper_agent: NewAgent,
        opponent: BasicAgent,
        target_ball: str = "solid",
        helper_player: str = "A",
        max_steps: int = 100,
        disable_env_noise: bool = True,
    ) -> None:
        super().__init__()
        self.helper = helper_agent
        self.opponent = opponent
        self.target_ball = target_ball
        if helper_player not in {"A", "B"}:
            raise ValueError(f"helper_player must be 'A' or 'B', got {helper_player}")
        self.helper_player = helper_player
        self.opponent_player = "B" if helper_player == "A" else "A"
        self.max_steps = max_steps

        self.pool_env = PoolEnv()
        self.pool_env.enable_noise = not disable_env_noise

        self.action_space = gym.spaces.Box(
            low=self.helper.action_low,
            high=self.helper.action_high,
            dtype=np.float32,
        )

        self._episode_reward = 0.0
        self._episode_length = 0
        self._max_reset_attempts = 10

        # 预先重置以确定观测维度
        self.pool_env.reset(target_ball=self.target_ball)
        initial_obs = self._encode_current_observation()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )

    def _encode_current_observation(self) -> np.ndarray:
        balls, my_targets, table = self.pool_env.get_observation(self.helper_player)
        return self.helper._encode_observation(balls, my_targets, table)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and "target_ball" in options:
            self.target_ball = options["target_ball"]
        attempts = 0
        while True:
            self.pool_env.reset(target_ball=self.target_ball)
            self._episode_reward = 0.0
            self._episode_length = 0
            self._advance_to_helper_turn()
            terminated, _ = self.pool_env.get_done()
            if (not terminated) and self.pool_env.get_curr_player() == self.helper_player:
                break
            attempts += 1
            if attempts >= self._max_reset_attempts:
                raise RuntimeError(
                    "Failed to initialise environment: helper player never obtained turn after reset."
                )
        observation = self._encode_current_observation()
        return observation.astype(np.float32), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.pool_env.get_curr_player() != self.helper_player:
            raise RuntimeError(
                "step called when it is not the helper player's turn. Ensure observations are consumed correctly."
            )
        balls, my_targets, table = self.pool_env.get_observation(self.helper_player)
        pre_shot_state = save_balls_state(self.pool_env.balls)
        formatted_action = self.helper._format_action(action)

        self.pool_env.take_shot(formatted_action)
        shot = self.pool_env.shot_record[-1] if len(self.pool_env.shot_record) > 0 else None
        reward = 0.0
        if shot is not None:
            reward = analyze_shot_for_reward(shot, pre_shot_state, my_targets)

        self._episode_reward += reward
        self._episode_length += 1

        terminated, info = self.pool_env.get_done()
        truncated = self._episode_length >= self.max_steps

        # 如果对方回合，执行对方动作直到轮到 A 或比赛结束
        if (not terminated) and (not truncated):
            self._advance_to_helper_turn()
            terminated, info = self.pool_env.get_done()

        observation = self._encode_current_observation()

        if truncated and not terminated:
            info = info or {}
            info["truncated"] = True

        info = info or {}
        info.update({
            "episode_reward": self._episode_reward,
            "episode_length": self._episode_length,
        })

        return observation.astype(np.float32), float(reward), terminated, truncated, info

    def _advance_to_helper_turn(self) -> None:
        terminated, _ = self.pool_env.get_done()
        while (not terminated) and self.pool_env.get_curr_player() != self.helper_player:
            self._opponent_take_turn()
            terminated, _ = self.pool_env.get_done()

    def _opponent_take_turn(self) -> None:
        opp_balls, opp_targets, opp_table = self.pool_env.get_observation(self.opponent_player)
        opp_action = self.opponent.decision(opp_balls, opp_targets, opp_table)
        self.pool_env.take_shot(opp_action)


class TrainStatsCallback(BaseCallback):
    """记录训练统计数据并处理周期性保存。"""

    def __init__(
        self,
        save_every: int,
        checkpoint_dir: str,
        writer: Optional[SummaryWriter],
        log_interval: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer
        self.log_interval = max(1, log_interval)
        self.episode = 0
        self.recent_rewards: deque[float] = deque(maxlen=self.log_interval)
        self.best_mean_reward: Optional[float] = None
        self.best_model_episode: Optional[int] = None
        self._best_model_tmp_path = os.path.join(self.checkpoint_dir, "_best_model_tmp.zip")
        if os.path.exists(self._best_model_tmp_path):
            os.remove(self._best_model_tmp_path)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones") or []
        infos = self.locals.get("infos") or []
        for done, info in zip(dones, infos):
            if done:
                self.episode += 1
                ep_reward = info.get("episode_reward", 0.0)
                ep_length = info.get("episode_length", 0)
                self.recent_rewards.append(ep_reward)

                if self.writer is not None:
                    self.writer.add_scalar("episode/total_reward", ep_reward, self.episode)
                    self.writer.add_scalar("episode/length", ep_length, self.episode)

                mean_span = len(self.recent_rewards)
                mean_reward = float(np.mean(self.recent_rewards)) if mean_span > 0 else 0.0
                if self.writer is not None and mean_span == self.recent_rewards.maxlen:
                    self.writer.add_scalar("episode/mean_reward", mean_reward, self.episode)

                if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_model_episode = self.episode
                    if self.model is not None:
                        self.model.save(self._best_model_tmp_path)
                    if self.verbose:
                        print(
                            f"New best mean reward {mean_reward:.2f} at episode {self.episode:04d}"
                        )

                if self.verbose:
                    print(
                        f"[Episode {self.episode:04d}] reward={ep_reward:.2f}, length={ep_length}, "
                        f"mean_reward(last {mean_span})={mean_reward:.2f}"
                    )

                if self.save_every > 0 and self.episode % self.save_every == 0:
                    ckpt_path = os.path.join(
                        self.checkpoint_dir,
                        f"ppo_newagent_ep{self.episode:04d}.zip",
                    )
                    self.model.save(ckpt_path)
                    if self.verbose:
                        print(f"Saved checkpoint to {ckpt_path}")

        return True

    def _on_training_end(self) -> None:
        if self.best_mean_reward is not None and self.best_model_episode is not None:
            final_best_path = os.path.join(
                self.checkpoint_dir,
                f"best_model_ep{self.best_model_episode:04d}.zip",
            )
            if os.path.exists(self._best_model_tmp_path):
                os.replace(self._best_model_tmp_path, final_best_path)
            elif self.model is not None:
                self.model.save(final_best_path)
            if self.verbose:
                print(
                    f"Best mean reward {self.best_mean_reward:.2f} achieved at episode {self.best_model_episode:04d}. "
                    f"Saved best model to {final_best_path}"
                )
        elif self.verbose:
            print("Training completed but no episodes finished; best model not saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO NewAgent in PoolEnv")
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--player-role", choices=["A", "B"], default="A",
                        help="Player identity controlled by NewAgent during training")
    parser.add_argument("--ball-type", choices=["solid", "stripe"], default="solid",
                        help="Target ball type assigned to NewAgent for this training run")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum turns (shots) per episode to avoid stalls")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episode interval for logging aggregated statistics")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Episode interval for checkpointing. Set 0 to disable")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to store model checkpoints")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Optional PPO checkpoint (.zip) to resume training from")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device string, e.g. 'cuda:0'. Defaults to auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-env-noise", action="store_true",
                        help="Disable environment shot noise during training")
    parser.add_argument("--verbose", action="store_true", help="Print PPO update metrics")
    parser.add_argument("--tb-logdir", type=str, default="runs/ppo",
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
    training_agent = NewAgent(device=args.device, deterministic_eval=False, training=True)

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
        f"Training NewAgent as Player {player_role} with {ball_type} balls. "
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
    print(f"Starting training for {total_timesteps} timesteps.")
    training_agent.learn(total_timesteps=total_timesteps, callback=callback)

    if writer is not None:
        writer.close()

    train_env.close()


if __name__ == "__main__":
    main()
