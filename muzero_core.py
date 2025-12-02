"""
muzero_core.py - MuZero核心网络架构

实现MuZero的三个核心网络：
1. Representation Network (h): s -> hidden_state
2. Dynamics Network (g): (hidden_state, action) -> (next_hidden_state, reward)
3. Prediction Network (f): hidden_state -> (policy, value)

适配台球环境的连续动作空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class RepresentationNetwork(nn.Module):
    """表示网络：将观测编码为隐状态

    输入：球的位置、速度、目标球信息
    输出：隐状态向量
    """

    def __init__(self,
                 state_dim: int = 128,
                 hidden_dim: int = 256):
        """
        参数：
            state_dim: 隐状态维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 输入特征维度
        # 每个球：位置(x,y) + 速度(vx,vy) + 是否己方目标球(1) = 5维
        # 最多16个球 -> 16*5 = 80维
        # 额外：白球位置(2) + 剩余目标球数(1) = 3维
        # 总计：83维
        input_dim = 83

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()  # 将隐状态归一化到[-1, 1]
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        参数：
            observation: [batch_size, 83] 球的状态特征

        返回：
            hidden_state: [batch_size, state_dim] 隐状态
        """
        return self.encoder(observation)


class DynamicsNetwork(nn.Module):
    """动力学网络：预测状态转移和即时奖励

    学习环境的物理模拟（替代慢速的pooltool物理引擎）
    """

    def __init__(self,
                 state_dim: int = 128,
                 action_dim: int = 5,
                 hidden_dim: int = 256):
        """
        参数：
            state_dim: 隐状态维度
            action_dim: 动作维度 (V0, phi, theta, a, b)
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 动作归一化层
        self.action_normalizer = nn.Linear(action_dim, 32)

        # 状态转移网络
        self.transition = nn.Sequential(
            nn.Linear(state_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()
        )

        # 奖励预测网络
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 预测标量奖励
        )

    def forward(self,
                hidden_state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            hidden_state: [batch_size, state_dim]
            action: [batch_size, 5] 归一化的动作

        返回：
            next_hidden_state: [batch_size, state_dim]
            reward: [batch_size, 1]
        """
        # 编码动作
        action_encoded = F.relu(self.action_normalizer(action))

        # 拼接状态和动作
        state_action = torch.cat([hidden_state, action_encoded], dim=-1)

        # 预测下一状态
        next_hidden_state = self.transition(state_action)

        # 预测即时奖励
        reward = self.reward_head(state_action)

        return next_hidden_state, reward


class PredictionNetwork(nn.Module):
    """预测网络：预测策略（动作分布）和价值

    对于连续动作空间，输出高斯分布的参数(μ, σ)
    """

    def __init__(self,
                 state_dim: int = 128,
                 action_dim: int = 5,
                 hidden_dim: int = 256):
        """
        参数：
            state_dim: 隐状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.action_dim = action_dim

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 策略头：输出动作分布的均值和标准差
        self.policy_mu = nn.Linear(hidden_dim, action_dim)
        self.policy_sigma = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # 确保标准差为正
        )

        # 价值头：评估当前局面
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数：
            hidden_state: [batch_size, state_dim]

        返回：
            policy_mu: [batch_size, action_dim] 动作均值
            policy_sigma: [batch_size, action_dim] 动作标准差
            value: [batch_size, 1] 状态价值
        """
        features = self.shared(hidden_state)

        # 策略分布参数
        mu = self.policy_mu(features)
        sigma = self.policy_sigma(features) + 1e-3  # 添加小常数避免0

        # 价值
        value = self.value_head(features)

        return mu, sigma, value


class MuZeroNetwork(nn.Module):
    """MuZero完整网络：整合三个核心网络"""

    def __init__(self,
                 state_dim: int = 128,
                 action_dim: int = 5,
                 hidden_dim: int = 256):
        """
        参数：
            state_dim: 隐状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.representation = RepresentationNetwork(state_dim, hidden_dim)
        self.dynamics = DynamicsNetwork(state_dim, action_dim, hidden_dim)
        self.prediction = PredictionNetwork(state_dim, action_dim, hidden_dim)

        # 动作归一化参数（根据台球动作范围）
        self.action_mean = torch.tensor([4.25, 180.0, 45.0, 0.0, 0.0])
        self.action_std = torch.tensor([3.75, 180.0, 45.0, 0.5, 0.5])

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """归一化动作到[-1, 1]范围"""
        return (action - self.action_mean.to(action.device)) / self.action_std.to(action.device)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """反归一化动作到原始范围"""
        return action * self.action_std.to(action.device) + self.action_mean.to(action.device)

    def initial_inference(self, observation: torch.Tensor) -> Dict:
        """初始推理：从观测开始

        参数：
            observation: [batch_size, 83]

        返回：
            dict: {
                'hidden_state': 隐状态,
                'policy_mu': 策略均值,
                'policy_sigma': 策略标准差,
                'value': 价值
            }
        """
        hidden_state = self.representation(observation)
        policy_mu, policy_sigma, value = self.prediction(hidden_state)

        return {
            'hidden_state': hidden_state,
            'policy_mu': policy_mu,
            'policy_sigma': policy_sigma,
            'value': value
        }

    def recurrent_inference(self,
                           hidden_state: torch.Tensor,
                           action: torch.Tensor) -> Dict:
        """递归推理：从隐状态和动作预测下一步

        参数：
            hidden_state: [batch_size, state_dim]
            action: [batch_size, 5] 原始动作

        返回：
            dict: {
                'hidden_state': 下一隐状态,
                'reward': 即时奖励,
                'policy_mu': 策略均值,
                'policy_sigma': 策略标准差,
                'value': 价值
            }
        """
        # 归一化动作
        action_norm = self.normalize_action(action)

        # 动力学预测
        next_hidden_state, reward = self.dynamics(hidden_state, action_norm)

        # 策略和价值预测
        policy_mu, policy_sigma, value = self.prediction(next_hidden_state)

        return {
            'hidden_state': next_hidden_state,
            'reward': reward,
            'policy_mu': policy_mu,
            'policy_sigma': policy_sigma,
            'value': value
        }


def encode_observation(balls: Dict, my_targets: List[str], table) -> np.ndarray:
    """将台球环境的观测编码为网络输入

    参数：
        balls: {ball_id: Ball} 球状态字典
        my_targets: 己方目标球ID列表
        table: 球桌对象

    返回：
        observation: [83] 特征向量
    """
    feature_vector = []

    # 标准化位置（球桌尺寸约2m x 1m）
    table_width = table.w if hasattr(table, 'w') else 1.0
    table_length = table.l if hasattr(table, 'l') else 2.0

    # 为每个球编码特征（按固定顺序）
    ball_ids = ['cue'] + [str(i) for i in range(1, 8)] + ['8'] + [str(i) for i in range(9, 16)]

    for ball_id in ball_ids:
        if ball_id in balls:
            ball = balls[ball_id]

            # 位置 (归一化到[0, 1])
            if ball.state.s == 4:  # 已进袋
                x, y = -1.0, -1.0  # 特殊标记
            else:
                x = ball.state.rvw[0][0] / table_length
                y = ball.state.rvw[0][1] / table_width

            # 速度 (通常很小，归一化到合理范围)
            vx = np.tanh(ball.state.rvw[1][0])
            vy = np.tanh(ball.state.rvw[1][1])

            # 是否是己方目标球
            is_target = 1.0 if ball_id in my_targets else 0.0

            feature_vector.extend([x, y, vx, vy, is_target])
        else:
            # 球不存在（已进袋或其他原因）
            feature_vector.extend([-1.0, -1.0, 0.0, 0.0, 0.0])

    # 额外全局特征
    # 白球位置
    if 'cue' in balls and balls['cue'].state.s != 4:
        cue_x = balls['cue'].state.rvw[0][0] / table_length
        cue_y = balls['cue'].state.rvw[0][1] / table_width
    else:
        cue_x, cue_y = 0.0, 0.0

    # 剩余目标球数（归一化到[0, 1]）
    remaining_targets = sum(1 for tid in my_targets if tid in balls and balls[tid].state.s != 4)
    remaining_ratio = remaining_targets / 7.0  # 最多7个目标球

    feature_vector.extend([cue_x, cue_y, remaining_ratio])

    return np.array(feature_vector, dtype=np.float32)


if __name__ == '__main__':
    """测试网络"""
    print("=== 测试MuZero网络 ===")

    # 创建网络
    muzero_net = MuZeroNetwork(
        state_dim=128,
        action_dim=5,
        hidden_dim=256
    )

    print(f"网络参数量: {sum(p.numel() for p in muzero_net.parameters()):,}")

    # 测试初始推理
    batch_size = 4
    obs = torch.randn(batch_size, 83)

    result = muzero_net.initial_inference(obs)
    print(f"\n初始推理:")
    print(f"  隐状态形状: {result['hidden_state'].shape}")
    print(f"  策略均值: {result['policy_mu'].shape}")
    print(f"  策略标准差: {result['policy_sigma'].shape}")
    print(f"  价值: {result['value'].shape}")

    # 测试递归推理
    action = torch.tensor([[5.0, 45.0, 0.0, 0.0, 0.0]] * batch_size)

    result = muzero_net.recurrent_inference(result['hidden_state'], action)
    print(f"\n递归推理:")
    print(f"  下一隐状态形状: {result['hidden_state'].shape}")
    print(f"  奖励: {result['reward'].shape}")
    print(f"  价值: {result['value'].shape}")

    print("\n✓ 网络测试通过！")
