"""
muzero_trainer.py - MuZero训练器

实现MuZero的训练循环，包括：
1. 自我对弈收集数据
2. 网络训练
3. 模型保存和加载
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import os
from datetime import datetime

from muzero_core import MuZeroNetwork
from muzero_replay import ReplayBuffer


class MuZeroTrainer:
    """MuZero训练器"""

    def __init__(self,
                 network: MuZeroNetwork,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 value_loss_weight: float = 1.0,
                 reward_loss_weight: float = 1.0,
                 policy_loss_weight: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        参数：
            network: MuZero网络
            lr: 学习率
            weight_decay: 权重衰减
            value_loss_weight: 价值损失权重
            reward_loss_weight: 奖励损失权重
            policy_loss_weight: 策略损失权重
            device: 设备
        """
        self.network = network
        self.device = device
        self.network.to(device)

        # 损失权重
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight

        # 优化器
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9
        )

        # 统计信息
        self.train_steps = 0
        self.losses = {
            'total': [],
            'value': [],
            'reward': [],
            'policy': []
        }

    def train_batch(self,
                   observations: list,
                   actions_list: list,
                   targets_list: list) -> Dict[str, float]:
        """训练一个批次

        参数：
            observations: List of [83] 初始观测
            actions_list: List of List[5] 动作序列
            targets_list: List of Dict 目标值

        返回：
            losses: 各项损失
        """
        self.network.train()

        batch_size = len(observations)
        num_unroll_steps = len(actions_list[0]) - 1

        # 转换为tensor
        obs_batch = torch.FloatTensor(np.array(observations)).to(self.device)

        # 初始推理
        outputs = self.network.initial_inference(obs_batch)
        hidden_state = outputs['hidden_state']

        # 累积损失
        total_value_loss = 0.0
        total_reward_loss = 0.0
        total_policy_loss = 0.0

        # 计算初始步的损失（k=0）
        pred_value = outputs['value']
        pred_mu = outputs['policy_mu']
        pred_sigma = outputs['policy_sigma']

        # 目标值
        target_values = torch.FloatTensor([t['value'][0] for t in targets_list]).unsqueeze(1).to(self.device)
        target_mu = torch.FloatTensor([t['policy_mu'][0] for t in targets_list]).to(self.device)
        target_sigma = torch.FloatTensor([t['policy_sigma'][0] for t in targets_list]).to(self.device)

        # 价值损失（MSE）
        value_loss = F.mse_loss(pred_value, target_values)
        total_value_loss += value_loss

        # 策略损失（负对数似然 + KL散度）
        policy_loss = self._policy_loss(pred_mu, pred_sigma, target_mu, target_sigma)
        total_policy_loss += policy_loss

        # 展开步的损失（k=1 to num_unroll_steps）
        for k in range(num_unroll_steps):
            # 获取动作
            actions = torch.FloatTensor([actions_list[i][k] for i in range(batch_size)]).to(self.device)

            # 递归推理
            outputs = self.network.recurrent_inference(hidden_state, actions)
            hidden_state = outputs['hidden_state']
            pred_reward = outputs['reward']
            pred_value = outputs['value']
            pred_mu = outputs['policy_mu']
            pred_sigma = outputs['policy_sigma']

            # 目标值
            target_rewards = torch.FloatTensor([t['reward'][k+1] for t in targets_list]).unsqueeze(1).to(self.device)
            target_values = torch.FloatTensor([t['value'][k+1] for t in targets_list]).unsqueeze(1).to(self.device)
            target_mu = torch.FloatTensor([t['policy_mu'][k+1] for t in targets_list]).to(self.device)
            target_sigma = torch.FloatTensor([t['policy_sigma'][k+1] for t in targets_list]).to(self.device)

            # 奖励损失
            reward_loss = F.mse_loss(pred_reward, target_rewards)
            total_reward_loss += reward_loss

            # 价值损失
            value_loss = F.mse_loss(pred_value, target_values)
            total_value_loss += value_loss

            # 策略损失
            policy_loss = self._policy_loss(pred_mu, pred_sigma, target_mu, target_sigma)
            total_policy_loss += policy_loss

        # 平均损失
        total_value_loss /= (num_unroll_steps + 1)
        total_reward_loss /= num_unroll_steps
        total_policy_loss /= (num_unroll_steps + 1)

        # 总损失
        total_loss = (
            self.value_loss_weight * total_value_loss +
            self.reward_loss_weight * total_reward_loss +
            self.policy_loss_weight * total_policy_loss
        )

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # 记录
        self.train_steps += 1
        losses = {
            'total': total_loss.item(),
            'value': total_value_loss.item(),
            'reward': total_reward_loss.item(),
            'policy': total_policy_loss.item()
        }

        for key, value in losses.items():
            self.losses[key].append(value)

        return losses

    def _policy_loss(self,
                    pred_mu: torch.Tensor,
                    pred_sigma: torch.Tensor,
                    target_mu: torch.Tensor,
                    target_sigma: torch.Tensor) -> torch.Tensor:
        """计算策略损失（高斯分布的KL散度）

        参数：
            pred_mu: [batch, 5] 预测均值
            pred_sigma: [batch, 5] 预测标准差
            target_mu: [batch, 5] 目标均值
            target_sigma: [batch, 5] 目标标准差

        返回：
            loss: 标量损失
        """
        # KL散度：KL(target || pred)
        # KL(N(μ1,σ1) || N(μ2,σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2

        kl_div = (
            torch.log(pred_sigma / (target_sigma + 1e-8)) +
            (target_sigma ** 2 + (target_mu - pred_mu) ** 2) / (2 * pred_sigma ** 2 + 1e-8) -
            0.5
        )

        return kl_div.mean()

    def train_epoch(self, replay_buffer: ReplayBuffer, num_batches: int = 100) -> Dict[str, float]:
        """训练一个epoch

        参数：
            replay_buffer: 重放缓冲区
            num_batches: 批次数

        返回：
            avg_losses: 平均损失
        """
        epoch_losses = {'total': 0.0, 'value': 0.0, 'reward': 0.0, 'policy': 0.0}

        for _ in range(num_batches):
            # 采样批次
            observations, actions_list, targets_list = replay_buffer.sample_batch()

            # 训练
            losses = self.train_batch(observations, actions_list, targets_list)

            # 累积
            for key in epoch_losses:
                epoch_losses[key] += losses[key]

        # 平均
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def save_checkpoint(self, filepath: str, epoch: int, replay_buffer: ReplayBuffer = None):
        """保存检查点

        参数：
            filepath: 保存路径
            epoch: 当前epoch
            replay_buffer: 可选的重放缓冲区
        """
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'losses': self.losses
        }

        torch.save(checkpoint, filepath)
        print(f"✓ 检查点已保存到 {filepath}")

        # 保存重放缓冲区
        if replay_buffer is not None:
            buffer_path = filepath.replace('.pt', '_buffer.pkl')
            replay_buffer.save_to_disk(buffer_path)

    def load_checkpoint(self, filepath: str, replay_buffer: ReplayBuffer = None) -> int:
        """加载检查点

        参数：
            filepath: 检查点路径
            replay_buffer: 可选的重放缓冲区

        返回：
            epoch: 加载的epoch
        """
        if not os.path.exists(filepath):
            print(f"检查点文件 {filepath} 不存在")
            return 0

        checkpoint = torch.load(filepath, map_location=self.device)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_steps = checkpoint['train_steps']
        self.losses = checkpoint['losses']

        print(f"✓ 从 {filepath} 加载检查点 (epoch {checkpoint['epoch']})")

        # 加载重放缓冲区
        if replay_buffer is not None:
            buffer_path = filepath.replace('.pt', '_buffer.pkl')
            replay_buffer.load_from_disk(buffer_path)

        return checkpoint['epoch']


# 为了避免循环导入，这里只导入需要的
import torch.nn.functional as F


def test_trainer():
    """测试训练器"""
    print("=== 测试训练器 ===")

    # 创建网络
    network = MuZeroNetwork(state_dim=128, action_dim=5, hidden_dim=256)

    # 创建训练器
    trainer = MuZeroTrainer(
        network=network,
        lr=3e-4,
        device='cpu'  # 测试用CPU
    )

    # 创建模拟批次
    batch_size = 4
    num_unroll_steps = 5

    observations = [np.random.randn(83) for _ in range(batch_size)]
    actions_list = [[np.random.randn(5) for _ in range(num_unroll_steps + 1)]
                   for _ in range(batch_size)]

    targets_list = []
    for _ in range(batch_size):
        targets = {
            'value': [np.random.randn() for _ in range(num_unroll_steps + 1)],
            'reward': [np.random.randn() for _ in range(num_unroll_steps + 1)],
            'policy_mu': [np.random.randn(5) for _ in range(num_unroll_steps + 1)],
            'policy_sigma': [np.abs(np.random.randn(5)) + 0.1 for _ in range(num_unroll_steps + 1)]
        }
        targets_list.append(targets)

    # 训练一个批次
    losses = trainer.train_batch(observations, actions_list, targets_list)

    print(f"训练步数: {trainer.train_steps}")
    print(f"损失:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ 训练器测试通过！")


if __name__ == '__main__':
    test_trainer()
