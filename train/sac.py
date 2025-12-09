import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
    

BALL_ORDER: List[str] = ['cue'] + [str(i) for i in range(1, 16)]
ACTION_KEYS: List[str] = ['V0', 'phi', 'theta', 'a', 'b']
ACTION_BOUNDS: Dict[str, Tuple[float, float]] = {
    'V0': (0.5, 8.0),
    'phi': (0.0, 360.0),
    'theta': (0.0, 90.0),
    'a': (-0.5, 0.5),
    'b': (-0.5, 0.5),
}

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _scale_action_component(action_value: float, low_high: Tuple[float, float]) -> float:
    """将 [-1,1] 动作值映射为环境实际范围"""
    low, high = low_high
    return low + (action_value + 1.0) * 0.5 * (high - low)


def _normalize_action_component(value: float, low_high: Tuple[float, float]) -> float:
    """将实际动作范围映射为 [-1,1]"""
    low, high = low_high
    return (2.0 * (value - low) / (high - low)) - 1.0


@dataclass
class SACConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    automatic_entropy_tuning: bool = True
    policy_update_freq: int = 1


class ReplayBuffer:
    """简单的循环式经验回放缓存"""

    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )
        return batch

    def __len__(self):
        return self.size


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action




class SACAgent():
    """基于 Soft Actor-Critic 的台球 Agent"""

    def __init__(self, config: Optional[SACConfig] = None, checkpoint_path: str = "checkpoints/sac_agent.pth", training: bool = False):
        super().__init__()

        self.config = config or SACConfig()
        self.checkpoint_path = Path(checkpoint_path)
        self.training_enabled = training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = self._compute_state_dim()
        self.action_dim = len(ACTION_KEYS)

        # 网络与优化器
        self.actor = GaussianPolicy(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.critic1 = SoftQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.critic2 = SoftQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.critic1_target = SoftQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.critic2_target = SoftQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.lr_critic)

        self.automatic_entropy_tuning = self.config.automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(self.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(self.config.alpha, device=self.device)
            self.log_alpha = None
            self.alpha_optim = None

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.config.buffer_size) if self.training_enabled else None
        self.gradient_updates = 0

        # 若存在已训练的权重则加载
        self.load_checkpoint()

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        state = self._encode_state(balls, my_targets, table)
        action, _ = self._act(state, evaluate=not self.training_enabled)
        return action

    # --------------------------- 推理与训练辅助函数 --------------------------- #

    def _compute_state_dim(self) -> int:
        per_ball_features = 6  # x, y, vx, vy, pocketed, target-flag
        extra_features = 2     # 剩余目标比率、是否打黑8阶段
        return len(BALL_ORDER) * per_ball_features + extra_features

    def _encode_state(self, balls: dict, my_targets: List[str], table: Table) -> np.ndarray:
        features: List[float] = []
        table_l = getattr(table, 'l', 2.84) or 2.84
        table_w = getattr(table, 'w', 1.42) or 1.42
        for bid in BALL_ORDER:
            ball = balls.get(bid)
            if ball is None:
                features.extend([0.0] * 6)
                continue
            pos = ball.state.rvw[0]
            vel = ball.state.rvw[1]
            pocketed = 1.0 if ball.state.s == 4 else 0.0
            target_flag = 1.0 if bid in my_targets else 0.0
            features.extend([
                float(pos[0] / table_l),
                float(pos[1] / table_w),
                float(vel[0]),
                float(vel[1]),
                pocketed,
                target_flag,
            ])

        remaining_targets = sum(1 for bid in my_targets if bid in balls and balls[bid].state.s != 4 and bid != '8')
        total_targets = max(1, sum(1 for bid in my_targets if bid != '8'))
        phase_black = 1.0 if (len(my_targets) == 1 and my_targets[0] == '8') else 0.0
        features.append(remaining_targets / total_targets)
        features.append(phase_black)
        return np.asarray(features, dtype=np.float32)

    def encode_observation(self, balls: dict, my_targets: List[str], table: Table) -> np.ndarray:
        """公开的观测编码接口，便于训练脚本复用"""
        return self._encode_state(balls, my_targets, table)

    def _random_action(self) -> Dict[str, float]:
        """回退策略：随机动作"""
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3),
        }

    def _act(self, state: np.ndarray, evaluate: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, mean_action = self.actor.sample(state_tensor)
                scaled_action = mean_action.cpu().numpy()[0]
            else:
                sampled_action, _, _ = self.actor.sample(state_tensor)
                scaled_action = sampled_action.cpu().numpy()[0]
        action_dict = self._scaled_action_to_dict(scaled_action)
        return action_dict, scaled_action

    def _scaled_action_to_dict(self, scaled_action: np.ndarray) -> Dict[str, float]:
        action: Dict[str, float] = {}
        for idx, key in enumerate(ACTION_KEYS):
            mapped_value = _scale_action_component(float(scaled_action[idx]), ACTION_BOUNDS[key])
            if key == 'phi':
                mapped_value %= 360.0
            low, high = ACTION_BOUNDS[key]
            action[key] = float(np.clip(mapped_value, low, high))
        return action

    def _dict_to_scaled_action(self, action: Dict[str, float]) -> np.ndarray:
        scaled = [
            _normalize_action_component(float(action[key]), ACTION_BOUNDS[key])
            for key in ACTION_KEYS
        ]
        return np.asarray(scaled, dtype=np.float32)

    def store_transition(self, state: np.ndarray, action: Dict[str, float], reward: float, next_state: np.ndarray, done: bool):
        if not self.training_enabled or self.replay_buffer is None:
            return
        scaled_action = self._dict_to_scaled_action(action)
        self.replay_buffer.add(state, scaled_action, [reward], next_state, [float(done)])

    def update_parameters(self):
        if not self.training_enabled or self.replay_buffer is None:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch['dones'], dtype=torch.float32, device=self.device)

        if isinstance(self.alpha, torch.Tensor):
            alpha_value = self.alpha.detach()
        else:
            alpha_value = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_action)
            q2_next = self.critic2_target(next_states, next_action)
            min_q_next = torch.min(q1_next, q2_next) - alpha_value * next_log_prob
            q_target = rewards + (1 - dones) * self.config.gamma * min_q_next

        q1_pred = self.critic1(states, actions)
        q2_pred = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1_pred, q_target)
        critic2_loss = F.mse_loss(q2_pred, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha_value * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.gradient_updates += 1
        return {
            'critic1_loss': float(critic1_loss.item()),
            'critic2_loss': float(critic2_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha_loss': float(alpha_loss.item()) if alpha_loss is not None else None,
            'alpha': float(self.alpha.item()) if isinstance(self.alpha, torch.Tensor) else float(self.alpha),
        }

    def _soft_update(self, source_net, target_net):
        tau = self.config.tau
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_checkpoint(self, output_path: Optional[Path] = None):
        path = Path(output_path) if output_path is not None else self.checkpoint_path
        if not path:
            return
        payload = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'config': self.config,
        }
        if self.automatic_entropy_tuning:
            payload.update({
                'log_alpha': self.log_alpha,
                'alpha_optim': self.alpha_optim.state_dict(),
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self):
        if not self.checkpoint_path.is_file():
            return
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])

    @staticmethod
    def compute_dense_reward(step_info: dict, my_targets: List[str]) -> float:
        """
        计算密集奖励函数

        设计理念：
        - 鼓励进攻：进球获得正奖励，无效击球给予小惩罚
        - 严厉惩罚犯规：白球进袋、非法击球等
        - 强化胜负：合法黑8给予大奖励，非法黑8严重惩罚
        - 进度激励：剩余球越少，每球价值越高
        """
        reward = 0.0

        # 1. 进球奖励（基础分 + 进度加成）
        my_pocketed = step_info.get('ME_INTO_POCKET', [])
        if my_pocketed:
            # 计算剩余目标球数（不含黑8）
            remaining_targets = len([t for t in my_targets if t != '8'])
            # 基础奖励50分，随着剩余球减少，奖励增加（最后一颗最多75分）
            progress_bonus = 1.0 + (7 - remaining_targets) * 0.05  # 1.0 到 1.35
            reward += 50.0 * len(my_pocketed) * progress_bonus

        # 2. 对方进球惩罚（虽然不常见，但可能因犯规导致）
        enemy_pocketed = step_info.get('ENEMY_INTO_POCKET', [])
        if enemy_pocketed:
            reward -= 25.0 * len(enemy_pocketed)

        # 3. 白球进袋（严重犯规）
        if step_info.get('WHITE_BALL_INTO_POCKET'):
            reward -= 100.0

        # 4. 黑8球进袋（胜负关键）
        if step_info.get('BLACK_BALL_INTO_POCKET'):
            # 合法打进黑8（己方目标球已清空）= 胜利
            legal = len(my_targets) == 1 and my_targets[0] == '8'
            if legal:
                reward += 300.0  # 大幅增加胜利奖励
            else:
                reward -= 300.0  # 大幅增加非法黑8惩罚

        # 5. 首球犯规（未先击中目标球）
        if step_info.get('FOUL_FIRST_HIT'):
            reward -= 30.0

        # 6. 无进球且无碰库（消极击球）
        if step_info.get('NO_POCKET_NO_RAIL'):
            reward -= 30.0

        # 7. 完全未击中任何球（严重失误）
        if step_info.get('NO_HIT'):
            reward -= 50.0

        # 8. 合法但无进球的击球：小惩罚，鼓励进攻而非保守
        if reward == 0.0:
            reward = -5.0  # 从 +10 改为 -5，鼓励主动进攻

        return float(reward)