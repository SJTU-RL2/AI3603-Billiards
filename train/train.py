from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT_DIR):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from poolenv import PoolEnv  
from sac import SACAgent, SACConfig  


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Soft Actor-Critic training for AI3603 billiards")
	parser.add_argument("--episodes", type=int, default=20, help="the number of episode")
	parser.add_argument("--control-player", type=str, choices=["A", "B"], default="A", help="the player controled by SAC agent")
	parser.add_argument("--target-cycle", type=str, default="solid,stripe", help="recyclable ball type, split by comma")
	parser.add_argument("--checkpoint", type=str, default="checkpoints/sac_agent.pth", help="checkpoint path")
	parser.add_argument("--log-dir", type=str, default="logs", help="log path")
	parser.add_argument("--save-every", type=int, default=5, help="how many echos we save the model")
	parser.add_argument("--selfplay-sync", type=int, default=5, help="episodes between syncing opponent weights")
	parser.add_argument("--seed", type=int, default=42, help="randon seed")
	parser.add_argument("--env-noise", action="store_true", help="use the environment noise")
	parser.add_argument("--learning-starts", type=int, default=512, help="start gradient update when buffer size reach this value")
	parser.add_argument("--updates-per-step", type=int, default=1, help="update times of the agent in a episodes")
	parser.add_argument("--hidden-dim", type=int, default=256, help="Actor/Critic hidden layer dimension")
	parser.add_argument("--batch-size", type=int, default=64, help="SAC batch size")
	parser.add_argument("--buffer-size", type=int, default=100_000, help="buffer size")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--tau", type=float, default=0.005, help="target network update soft factor")
	parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
	parser.add_argument("--lr-critic", type=float, default=3e-4, help="Critic learning rate")
	parser.add_argument("--lr-alpha", type=float, default=3e-4, help="alpha learning rate")
	parser.add_argument("--alpha", type=float, default=0.2, help="solid alpha (when forbidden auto entroy adjust)")
	parser.add_argument("--disable-auto-entropy", action="store_true", help="diable automatic entropy adjustment")
	return parser.parse_args()


def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except ImportError:
		pass



def rollout_opponent_turns(env: PoolEnv, opponent_agent: SACAgent, control_player: str) -> Tuple[float, bool]:
	"""让对手智能体连续出杆直到轮到训练智能体或对局结束"""
	cumulative_penalty = 0.0
	done, _ = env.get_done()
	while not done and env.get_curr_player() != control_player:
		player = env.get_curr_player()
		balls, my_targets, table = env.get_observation(player)
		state = opponent_agent.encode_observation(balls, my_targets, table)
		action_dict, _ = opponent_agent._act(state, evaluate=True)
		step_info = env.take_shot(action_dict)
		cumulative_penalty -= SACAgent.compute_dense_reward(step_info, my_targets)
		done, _ = env.get_done()
	return cumulative_penalty, done


def sync_opponent_agent(source: SACAgent, target: SACAgent) -> None:
	target.actor.load_state_dict(source.actor.state_dict())
	target.actor.eval()


def format_checkpoint_variant(base_path: Path, tag: str) -> Path:
	return base_path.with_name(f"{base_path.stem}_{tag}{base_path.suffix}")


def append_metrics(log_path: Path, row: dict) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	file_exists = log_path.exists()
	fieldnames = [
		"episode",
		"reward",
		"agent_turns",
		"buffer_size",
		"total_updates",
		"total_env_steps",
		"elapsed_sec",
	]
	with log_path.open("a", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerow(row)


def main():
	args = parse_args()
	set_global_seed(args.seed)

	target_cycle = [item.strip() for item in args.target_cycle.split(",") if item.strip()]
	if not target_cycle:
		raise ValueError("target-cycle 至少需要一个取值 (solid/stripe)")
	if args.save_every <= 0:
		raise ValueError("--save-every must be positive")
	if args.selfplay_sync <= 0:
		raise ValueError("--selfplay-sync must be positive")

	env = PoolEnv()
	env.enable_noise = args.env_noise
	checkpoint_base = Path(args.checkpoint)

	sac_config = SACConfig(
		hidden_dim=args.hidden_dim,
		gamma=args.gamma,
		tau=args.tau,
		alpha=args.alpha,
		lr_actor=args.lr_actor,
		lr_critic=args.lr_critic,
		lr_alpha=args.lr_alpha,
		batch_size=args.batch_size,
		buffer_size=args.buffer_size,
		automatic_entropy_tuning=not args.disable_auto_entropy,
	)

	sac_agent = SACAgent(
		config=sac_config,
		checkpoint_path=str(checkpoint_base),
		training=True,
	)
	opponent_checkpoint = format_checkpoint_variant(checkpoint_base, "opponent")
	opponent_agent = SACAgent(
		config=sac_config,
		checkpoint_path=str(opponent_checkpoint),
		training=False,
	)
	sync_opponent_agent(sac_agent, opponent_agent)
	log_path = Path(args.log_dir) / "training_metrics.csv"

	total_env_steps = 0
	total_updates = 0
	start_time = time.time()

	for episode in range(1, args.episodes + 1):
		target_ball = target_cycle[(episode - 1) % len(target_cycle)]
		env.reset(target_ball=target_ball)
		env.enable_noise = args.env_noise

		episode_reward = 0.0
		agent_turns = 0
		done, _ = env.get_done()

		if env.get_curr_player() != args.control_player:
			penalty, done = rollout_opponent_turns(env, opponent_agent, args.control_player)
			episode_reward += penalty

		while not done:
			if env.get_curr_player() != args.control_player:
				penalty, done = rollout_opponent_turns(env, opponent_agent, args.control_player)
				episode_reward += penalty
				if done:
					break

			balls, my_targets, table = env.get_observation(args.control_player)
			state = sac_agent.encode_observation(balls, my_targets, table)
			action_dict, _ = sac_agent._act(state, evaluate=False)

			step_info = env.take_shot(action_dict)
			immediate_reward = SACAgent.compute_dense_reward(step_info, my_targets)
			total_env_steps += 1

			done, _ = env.get_done()
			opponent_penalty = 0.0
			if not done:
				opponent_penalty, done = rollout_opponent_turns(env, opponent_agent, args.control_player)

			total_reward = immediate_reward + opponent_penalty
			episode_reward += total_reward

			if done:
				next_state = np.zeros_like(state)
			else:
				next_balls, next_targets, next_table = env.get_observation(args.control_player)
				next_state = sac_agent.encode_observation(next_balls, next_targets, next_table)

			sac_agent.store_transition(state, action_dict, total_reward, next_state, done)

			if sac_agent.replay_buffer and len(sac_agent.replay_buffer) >= args.learning_starts:
				for _ in range(args.updates_per_step):
					update_info = sac_agent.update_parameters()
					if update_info is not None:
						total_updates += 1

			agent_turns += 1

		if episode % args.save_every == 0:
			checkpoint_path = format_checkpoint_variant(checkpoint_base, f"ep{episode}")
			sac_agent.save_checkpoint(checkpoint_path)

		if episode % args.selfplay_sync == 0:
			sync_opponent_agent(sac_agent, opponent_agent)

		elapsed = time.time() - start_time
		buffer_size = len(sac_agent.replay_buffer) if sac_agent.replay_buffer else 0
		append_metrics(
			log_path,
			{
				"episode": episode,
				"reward": round(episode_reward, 2),
				"agent_turns": agent_turns,
				"buffer_size": buffer_size,
				"total_updates": total_updates,
				"total_env_steps": total_env_steps,
				"elapsed_sec": round(elapsed, 2),
			},
		)

		print(
			f"[Episode {episode}/{args.episodes}] reward={episode_reward:.1f} turns={agent_turns} "
			f"buffer={buffer_size} updates={total_updates} target={target_ball}"
		)

	sac_agent.save_checkpoint()
	print("Training finished. Checkpoints saved to", args.checkpoint)


if __name__ == "__main__":
	main()
