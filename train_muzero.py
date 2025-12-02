"""
train_muzero.py - MuZero训练脚本

完整的训练流程：
1. 自我对弈收集数据
2. 训练神经网络
3. 评估和保存模型
"""

import torch
import numpy as np
import os
from datetime import datetime
import argparse
import copy

from poolenv import PoolEnv
from muzero_core import MuZeroNetwork, encode_observation
from muzero_mcts import MCTS
from muzero_replay import ReplayBuffer, GameRecorder, compute_reward_from_step_info
from muzero_trainer import MuZeroTrainer


class MuZeroSelfPlay:
    """MuZero自我对弈"""

    def __init__(self,
                 network: MuZeroNetwork,
                 num_simulations: int = 30,
                 temperature: float = 1.0):
        """
        参数：
            network: MuZero网络
            num_simulations: MCTS模拟次数
            temperature: 温度参数
        """
        self.network = network
        self.mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            temperature=temperature
        )

    def play_game(self, env: PoolEnv, target_ball: str = 'solid') -> tuple:
        """进行一局自我对弈

        参数：
            env: 台球环境
            target_ball: 目标球型

        返回：
            (game_a, game_b): 双方的游戏记录
        """
        # 重置环境
        env.reset(target_ball=target_ball)

        # 创建记录器
        recorder_a = GameRecorder()
        recorder_b = GameRecorder()
        recorder_a.new_game('A')
        recorder_b.new_game('B')

        # 游戏循环
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)

            # 编码观测
            observation = encode_observation(balls, my_targets, table)

            # 保存击球前的状态
            balls_before = copy.deepcopy(balls)

            # MCTS搜索
            self.network.eval()
            with torch.no_grad():
                action_array = self.mcts.run(observation, add_noise=True)

                # 获取策略分布（用于训练）
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                result = self.network.initial_inference(obs_tensor)
                policy_mu = result['policy_mu'][0].cpu().numpy()
                policy_sigma = result['policy_sigma'][0].cpu().numpy()
                value = result['value'][0, 0].item()

            # 转换动作格式
            action = {
                'V0': float(action_array[0]),
                'phi': float(action_array[1]),
                'theta': float(action_array[2]),
                'a': float(action_array[3]),
                'b': float(action_array[4])
            }

            # 执行动作
            step_info = env.take_shot(action)

            # 计算奖励
            balls_after = step_info['BALLS']
            reward = compute_reward_from_step_info(
                step_info, my_targets, balls_before, balls_after
            )

            # 记录数据
            if player == 'A':
                recorder_a.record_step(
                    observation, action_array, reward,
                    (policy_mu, policy_sigma), value
                )
            else:
                recorder_b.record_step(
                    observation, action_array, reward,
                    (policy_mu, policy_sigma), value
                )

            # 检查游戏是否结束
            done, info = env.get_done()
            if done:
                winner = info['winner']
                hit_count = info['hit_count']

                game_a = recorder_a.end_game(winner, hit_count)
                game_b = recorder_b.end_game(winner, hit_count)

                return game_a, game_b


def train_muzero(config):
    """训练MuZero

    参数：
        config: 配置字典
    """
    print("=" * 60)
    print("MuZero台球训练")
    print("=" * 60)

    # 创建目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # 创建设备
    device = 'cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu'
    print(f"使用设备: {device}")

    # 创建网络
    network = MuZeroNetwork(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    )

    # 创建训练器
    trainer = MuZeroTrainer(
        network=network,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device
    )

    # 创建重放缓冲区
    replay_buffer = ReplayBuffer(
        max_size=config['replay_buffer_size'],
        batch_size=config['batch_size'],
        num_unroll_steps=config['num_unroll_steps'],
        td_steps=config['td_steps']
    )

    # 加载检查点（如果存在）
    start_epoch = 0
    if config['resume']:
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'latest.pt')
        start_epoch = trainer.load_checkpoint(checkpoint_path, replay_buffer)

    # 创建自我对弈器
    self_play = MuZeroSelfPlay(
        network=network,
        num_simulations=config['num_simulations'],
        temperature=config['temperature']
    )

    # 创建环境
    env = PoolEnv()

    print(f"\n配置:")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  自我对弈局数/轮: {config['games_per_epoch']}")
    print(f"  训练批次/轮: {config['batches_per_epoch']}")
    print(f"  MCTS模拟次数: {config['num_simulations']}")
    print(f"  批量大小: {config['batch_size']}")
    print()

    # 训练循环
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")

        # 1. 自我对弈收集数据
        print(f"\n[1/3] 自我对弈收集数据 ({config['games_per_epoch']} 局)...")
        target_balls = ['solid', 'stripe']

        for game_idx in range(config['games_per_epoch']):
            target_ball = target_balls[game_idx % 2]

            try:
                game_a, game_b = self_play.play_game(env, target_ball)

                # 保存到缓冲区
                replay_buffer.save_game(game_a)
                replay_buffer.save_game(game_b)

                print(f"  游戏 {game_idx + 1}/{config['games_per_epoch']}: "
                      f"胜者={game_a.winner}, 步数={len(game_a)}+{len(game_b)}, "
                      f"缓冲区大小={len(replay_buffer)}")

            except Exception as e:
                print(f"  游戏 {game_idx + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 2. 训练网络
        if len(replay_buffer) >= config['batch_size']:
            print(f"\n[2/3] 训练网络 ({config['batches_per_epoch']} 批次)...")

            losses = trainer.train_epoch(replay_buffer, config['batches_per_epoch'])

            print(f"  平均损失: total={losses['total']:.4f}, "
                  f"value={losses['value']:.4f}, "
                  f"reward={losses['reward']:.4f}, "
                  f"policy={losses['policy']:.4f}")
        else:
            print(f"\n[2/3] 跳过训练 (缓冲区大小 {len(replay_buffer)} < 批量大小 {config['batch_size']})")

        # 3. 保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            print(f"\n[3/3] 保存检查点...")

            # 保存最新模型
            latest_path = os.path.join(config['checkpoint_dir'], 'latest.pt')
            trainer.save_checkpoint(latest_path, epoch + 1, replay_buffer)

            # 保存编号模型
            epoch_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch + 1}.pt')
            trainer.save_checkpoint(epoch_path, epoch + 1)

        # 4. 评估（可选）
        if (epoch + 1) % config['eval_interval'] == 0 and config['eval_games'] > 0:
            print(f"\n[4/4] 评估模型...")
            eval_selfplay_against_basic(network, config['eval_games'])

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


def eval_selfplay_against_basic(network, num_games=10):
    """评估MuZero vs BasicAgent

    参数：
        network: MuZero网络
        num_games: 评估局数
    """
    from agent import BasicAgent

    env = PoolEnv()
    mcts = MCTS(network=network, num_simulations=30, temperature=0.0)
    basic_agent = BasicAgent()

    wins = {'muzero': 0, 'basic': 0, 'same': 0}

    for i in range(num_games):
        target_ball = ['solid', 'stripe'][i % 2]
        env.reset(target_ball=target_ball)

        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)

            # MuZero玩A，BasicAgent玩B
            if player == 'A':
                observation = encode_observation(balls, my_targets, table)
                network.eval()
                with torch.no_grad():
                    action_array = mcts.run(observation, add_noise=False)
                action = {
                    'V0': float(action_array[0]),
                    'phi': float(action_array[1]),
                    'theta': float(action_array[2]),
                    'a': float(action_array[3]),
                    'b': float(action_array[4])
                }
            else:
                action = basic_agent.decision(balls, my_targets, table)

            env.take_shot(action)

            done, info = env.get_done()
            if done:
                if info['winner'] == 'A':
                    wins['muzero'] += 1
                elif info['winner'] == 'B':
                    wins['basic'] += 1
                else:
                    wins['same'] += 1
                break

    print(f"  评估结果 ({num_games}局): MuZero={wins['muzero']}, "
          f"BasicAgent={wins['basic']}, 平局={wins['same']}")
    print(f"  MuZero胜率: {wins['muzero']/num_games*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练MuZero台球Agent')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--games_per_epoch', type=int, default=5, help='每轮自我对弈局数')
    parser.add_argument('--batches_per_epoch', type=int, default=50, help='每轮训练批次数')
    parser.add_argument('--save_interval', type=int, default=5, help='保存间隔')
    parser.add_argument('--eval_interval', type=int, default=10, help='评估间隔')
    parser.add_argument('--eval_games', type=int, default=10, help='评估局数')

    # 网络参数
    parser.add_argument('--state_dim', type=int, default=128, help='隐状态维度')
    parser.add_argument('--action_dim', type=int, default=5, help='动作维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')

    # MCTS参数
    parser.add_argument('--num_simulations', type=int, default=30, help='MCTS模拟次数')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--replay_buffer_size', type=int, default=500, help='重放缓冲区大小')
    parser.add_argument('--num_unroll_steps', type=int, default=5, help='展开步数')
    parser.add_argument('--td_steps', type=int, default=10, help='TD步数')

    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')

    args = parser.parse_args()

    # 转换为配置字典
    config = vars(args)

    # 开始训练
    train_muzero(config)
