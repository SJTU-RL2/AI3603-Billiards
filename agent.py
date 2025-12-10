"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

EPS = 1e-6


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


def _torch_atanh(x):
    if torch is None:
        raise ImportError("PyTorch is required for PPO agent.")
    x = x.clamp(-1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TanhNormal:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        std = torch.exp(torch.clamp(log_std, min=-20.0, max=2.0))
        self._normal = Normal(mean, std)

    def sample(self):
        pre_tanh = self._normal.rsample()
        return torch.tanh(pre_tanh), pre_tanh

    def log_prob(self, value: torch.Tensor, pre_tanh: Optional[torch.Tensor] = None):
        value = value.clamp(-1.0 + EPS, 1.0 - EPS)
        if pre_tanh is None:
            pre_tanh = _torch_atanh(value)
        log_prob = self._normal.log_prob(pre_tanh) - torch.log(1.0 - value.pow(2) + EPS)
        return log_prob.sum(dim=-1)

    def entropy(self):
        return self._normal.entropy().sum(dim=-1)

    def mean_value(self):
        return torch.tanh(self._normal.mean)


@dataclass
class PPOConfig:
    rollout_size: int = 1024
    minibatch_size: int = 256
    ppo_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_sizes: Tuple[int, ...] = (256, 256)
    deterministic_eval: bool = False
    normalize_advantage: bool = True
    reward_scale: float = 1.0
    update_on_episode_end: bool = True


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.rewards)

    def add(self, obs: np.ndarray, action: np.ndarray, log_prob: float, reward: float, done: bool, value: float, next_value: float):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(1.0 if done else 0.0)
        self.values.append(value)
        self.next_values.append(next_value)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        if len(self.rewards) == 0:
            return
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            mask = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * next_values[idx] * mask - values[idx]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[idx] = gae
        self.advantages = advantages
        self.returns = advantages + values

    def get(self, device: torch.device):
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before get().")
        observations = torch.as_tensor(np.asarray(self.observations, dtype=np.float32), device=device)
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=device)
        log_probs = torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=device)
        returns = torch.as_tensor(self.returns, device=device)
        advantages = torch.as_tensor(self.advantages, device=device)
        values = torch.as_tensor(np.asarray(self.values, dtype=np.float32), device=device)
        return observations, actions, log_probs, returns, advantages, values


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        return torch.tanh(self.mean_head(features))


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        return self.value_head(features).squeeze(-1)

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """基于 PPO 的自定义 Agent"""

    def __init__(self, config: Optional[PPOConfig] = None, device: Optional[str] = None, training: bool = True):
        super().__init__()
        if torch is None or nn is None or optim is None or Normal is None:
            raise ImportError("NewAgent requires PyTorch. Please install torch>=1.10.")
        self.config = config or PPOConfig()
        if isinstance(self.config.hidden_sizes, list):
            self.config.hidden_sizes = tuple(self.config.hidden_sizes)
        self.training = training
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.buffer = RolloutBuffer()
        self.action_low = np.asarray([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.action_high = np.asarray([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
        self.action_scale = (self.action_high - self.action_low) * 0.5
        self.action_bias = self.action_low + self.action_scale
        self.action_dim = self.action_low.shape[0]
        self.actor: Optional[ActorNetwork] = None
        self.critic: Optional[CriticNetwork] = None
        self.optimizer: Optional["optim.Optimizer"] = None
        self.obs_dim: Optional[int] = None
        self._action_scale_tensor: Optional[torch.Tensor] = None
        self._action_bias_tensor: Optional[torch.Tensor] = None
        self._log_action_scale_sum: Optional[torch.Tensor] = None
        self._last_step_cache: Optional[Dict[str, np.ndarray]] = None
        self.total_updates = 0
        if self.training:
            print(f"NewAgent (PPO) 已初始化，使用设备: {self.device}")

    def set_training(self, training: bool):
        self.training = training
        if self.actor is not None:
            self.actor.train(training)
        if self.critic is not None:
            self.critic.train(training)

    def decision(self, balls=None, my_targets=None, table=None, deterministic: Optional[bool] = None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        observation = self._encode_observation(balls, my_targets, table)
        self._ensure_model_ready(observation)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        dist = self._get_policy_distribution(obs_tensor)
        if deterministic is None:
            use_deterministic = self.config.deterministic_eval and not self.training
        else:
            use_deterministic = deterministic
        with torch.no_grad():
            if use_deterministic:
                action_tanh = dist.mean_value().squeeze(0)
                pre_tanh = _torch_atanh(action_tanh)
            else:
                sampled, pre_tanh = dist.sample()
                action_tanh = sampled.squeeze(0)
                pre_tanh = pre_tanh.squeeze(0)
            log_prob_tanh = dist.log_prob(action_tanh.unsqueeze(0), pre_tanh.unsqueeze(0))
            log_prob = (log_prob_tanh - self._log_action_scale_sum).item()
            value = float(self.critic(obs_tensor).item())
            scaled_tensor = self._scale_action_tensor(action_tanh)
        action_np = scaled_tensor.cpu().numpy()
        action_dict = self._format_action(action_np)
        if self.training:
            self._last_step_cache = {
                'observation': observation,
                'action_tanh': action_tanh.cpu().numpy(),
                'log_prob': log_prob,
                'value': value
            }
        return action_dict

    def observe(self, reward: float, done: bool, next_balls=None, next_targets=None, table=None):
        if not self.training or self.actor is None or self._last_step_cache is None:
            return None
        scaled_reward = float(reward) * self.config.reward_scale
        next_value = 0.0
        if not done and next_balls is not None and next_targets is not None and table is not None:
            next_observation = self._encode_observation(next_balls, next_targets, table)
            self._ensure_model_ready(next_observation)
            next_value = self._evaluate_value(next_observation)
        self.buffer.add(
            obs=self._last_step_cache['observation'],
            action=self._last_step_cache['action_tanh'],
            log_prob=self._last_step_cache['log_prob'],
            reward=scaled_reward,
            done=done,
            value=self._last_step_cache['value'],
            next_value=next_value
        )
        self._last_step_cache = None
        metrics = self._maybe_update(force=done)
        return metrics

    def _ensure_model_ready(self, observation: np.ndarray):
        obs_dim = observation.shape[0]
        if self.actor is not None and self.obs_dim == obs_dim:
            return
        self.obs_dim = obs_dim
        self.actor = ActorNetwork(obs_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.critic = CriticNetwork(obs_dim, self.config.hidden_sizes).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.config.lr, eps=1e-5)
        self._sync_action_tensors()
        self.actor.train(self.training)
        self.critic.train(self.training)

    def _sync_action_tensors(self):
        self._action_scale_tensor = torch.as_tensor(self.action_scale, dtype=torch.float32, device=self.device)
        self._action_bias_tensor = torch.as_tensor(self.action_bias, dtype=torch.float32, device=self.device)
        self._log_action_scale_sum = torch.log(self._action_scale_tensor).sum()

    def _get_policy_distribution(self, obs_tensor: torch.Tensor) -> TanhNormal:
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        mean = self.actor(obs_tensor)
        log_std = self.actor.log_std.expand_as(mean)
        return TanhNormal(mean, log_std)

    def _scale_action_tensor(self, action_tanh: torch.Tensor) -> torch.Tensor:
        return action_tanh * self._action_scale_tensor + self._action_bias_tensor

    def _evaluate_value(self, observation: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.critic(obs_tensor).item())

    def _maybe_update(self, force: bool = False):
        ready = len(self.buffer) >= self.config.rollout_size
        trigger = ready or (force and self.config.update_on_episode_end)
        if not trigger or len(self.buffer) == 0:
            return None
        metrics = self._update_policy()
        return metrics

    def update(self):
        if len(self.buffer) == 0:
            return None
        return self._update_policy()

    def _update_policy(self):
        self.buffer.compute_returns_and_advantages(self.config.gamma, self.config.gae_lambda)
        observations, actions, old_log_probs, returns, advantages, _ = self.buffer.get(self.device)
        if self.config.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data_size = observations.size(0)
        batch_size = min(self.config.minibatch_size, data_size)
        indices = np.arange(data_size)
        metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        total_batches = 0
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                obs_batch = observations[batch_idx]
                act_batch = actions[batch_idx]
                old_log_batch = old_log_probs[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]
                dist = self._get_policy_distribution(obs_batch)
                log_prob_tanh = dist.log_prob(act_batch)
                log_prob = log_prob_tanh - self._log_action_scale_sum
                entropy = dist.entropy() + self._log_action_scale_sum
                ratio = torch.exp(log_prob - old_log_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_pred = self.critic(obs_batch)
                if value_pred.dim() > 1:
                    value_pred = value_pred.squeeze(-1)
                value_loss = F.mse_loss(value_pred, ret_batch)
                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.config.max_grad_norm)
                self.optimizer.step()
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy_loss.item()
                total_batches += 1
        if total_batches > 0:
            for key in metrics:
                metrics[key] /= total_batches
        self.buffer.reset()
        self.total_updates += 1
        metrics['updates'] = self.total_updates
        metrics['samples'] = data_size
        return metrics

    def _encode_observation(self, balls, my_targets, table) -> np.ndarray:
        ball_ids = ['cue'] + [str(i) for i in range(1, 16)]
        target_set = set(my_targets or [])
        table_l = getattr(table, 'l', 1.0)
        table_w = getattr(table, 'w', 1.0)
        features: List[float] = []
        remaining_targets = 0
        for bid in ball_ids:
            ball = balls.get(bid)
            if ball is None:
                features.extend([0.0] * 10)
                continue
            state = ball.state
            pos = state.rvw[0]
            vel = state.rvw[1]
            pocketed = 1.0 if state.s == 4 else 0.0
            is_target = 1.0 if bid in target_set else 0.0
            if bid in target_set and pocketed == 0.0:
                remaining_targets += 1
            is_black = 1.0 if bid == '8' else 0.0
            features.extend([
                float(pos[0] / (table_l + EPS)),
                float(pos[1] / (table_w + EPS)),
                float(pos[2]),
                float(vel[0] / 5.0),
                float(vel[1] / 5.0),
                float(vel[2] / 5.0),
                float(state.s / 4.0),
                pocketed,
                is_target,
                is_black
            ])
        target_total = max(1, len(target_set))
        features.extend([
            len(target_set) / 7.0,
            remaining_targets / target_total,
            1.0 if target_set == {'8'} else 0.0,
            1.0
        ])
        return np.asarray(features, dtype=np.float32)

    def _format_action(self, action: np.ndarray) -> Dict[str, float]:
        clipped = np.clip(action, self.action_low, self.action_high)
        return {
            'V0': float(clipped[0]),
            'phi': float(clipped[1] % 360.0),
            'theta': float(clipped[2]),
            'a': float(clipped[3]),
            'b': float(clipped[4])
        }

    def save(self, filepath: str):
        if self.actor is None or self.critic is None:
            raise RuntimeError("Model not initialized; cannot save.")
        payload = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'config': self.config,
            'obs_dim': self.obs_dim
        }
        torch.save(payload, filepath)

    def load(self, filepath: str, map_location: Optional[str] = None):
        checkpoint = torch.load(filepath, map_location=map_location or self.device)
        obs_dim = checkpoint.get('obs_dim')
        if obs_dim is None:
            raise RuntimeError("Checkpoint missing observation dimension.")
        self.config = checkpoint.get('config', self.config)
        if isinstance(self.config.hidden_sizes, list):
            self.config.hidden_sizes = tuple(self.config.hidden_sizes)
        self._ensure_model_ready(np.zeros(obs_dim, dtype=np.float32))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if checkpoint.get('optimizer') is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.actor.train(self.training)
        self.critic.train(self.training)