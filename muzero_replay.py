"""
muzero_replay.py - 重放缓冲区和数据收集

实现经验重放缓冲区，用于存储和采样游戏数据
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple
from collections import deque
import random


class Game:
    """单局游戏的完整记录"""

    def __init__(self):
        """初始化游戏记录"""
        self.observations = []      # 观测序列
        self.actions = []           # 动作序列
        self.rewards = []           # 奖励序列
        self.policies = []          # 策略分布序列 (mu, sigma)
        self.values = []            # 价值序列

        self.winner = None          # 胜者 ('A', 'B', 'SAME')
        self.my_identity = None     # 我的身份 ('A' or 'B')
        self.hit_count = 0          # 总击球数

    def store_transition(self,
                        observation: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        policy: Tuple[np.ndarray, np.ndarray],
                        value: float):
        """存储一步转移

        参数：
            observation: [83] 观测
            action: [5] 动作
            reward: 标量奖励
            policy: (mu, sigma) 策略分布
            value: 价值估计
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)

    def terminal_reward(self) -> float:
        """计算终局奖励"""
        if self.winner == self.my_identity:
            return 100.0  # 获胜
        elif self.winner == 'SAME':
            return 0.0    # 平局
        else:
            return -100.0  # 失败

    def make_target(self,
                   state_index: int,
                   num_unroll_steps: int = 5,
                   td_steps: int = 10,
                   discount: float = 0.99) -> Dict:
        """为训练生成目标值

        参数：
            state_index: 当前状态索引
            num_unroll_steps: 展开步数
            td_steps: TD学习步数
            discount: 折扣因子

        返回：
            dict: 包含目标价值、奖励、策略
        """
        targets = {
            'value': [],
            'reward': [],
            'policy_mu': [],
            'policy_sigma': []
        }

        # 为每个展开步生成目标
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            # Bootstrap价值估计
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.values):
                value = self.values[bootstrap_index] * (discount ** td_steps)
            else:
                # 超出范围，使用终局奖励
                value = self.terminal_reward() * (discount ** (len(self.values) - current_index))

            # 累积中间奖励
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * (discount ** i)

            targets['value'].append(value)

            # 奖励目标（下一步的即时奖励）
            if current_index < len(self.rewards):
                targets['reward'].append(self.rewards[current_index])
            else:
                targets['reward'].append(0.0)

            # 策略目标
            if current_index < len(self.policies):
                mu, sigma = self.policies[current_index]
                targets['policy_mu'].append(mu)
                targets['policy_sigma'].append(sigma)
            else:
                # 填充零策略
                targets['policy_mu'].append(np.zeros(5))
                targets['policy_sigma'].append(np.ones(5))

        return targets

    def __len__(self):
        """游戏长度（步数）"""
        return len(self.observations)


class ReplayBuffer:
    """重放缓冲区"""

    def __init__(self,
                 max_size: int = 1000,
                 batch_size: int = 32,
                 num_unroll_steps: int = 5,
                 td_steps: int = 10):
        """
        参数：
            max_size: 最大存储游戏数
            batch_size: 批量大小
            num_unroll_steps: 每个样本的展开步数
            td_steps: TD学习步数
        """
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps

    def save_game(self, game: Game):
        """保存一局游戏"""
        if len(game) > 0:  # 确保游戏有数据
            self.buffer.append(game)

    def sample_batch(self) -> Tuple[List, List, List]:
        """采样一批训练数据

        返回：
            observations: List of [83]
            actions: List of List[5] (unroll_steps+1个动作)
            targets: List of Dict (包含value, reward, policy)
        """
        # 采样游戏
        games = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))

        observations = []
        actions_list = []
        targets_list = []

        for game in games:
            # 随机选择游戏中的一个位置
            game_pos = random.randint(0, len(game) - 1)

            # 初始观测
            observations.append(game.observations[game_pos])

            # 收集展开步的动作
            actions = []
            for step in range(self.num_unroll_steps + 1):
                if game_pos + step < len(game):
                    actions.append(game.actions[game_pos + step])
                else:
                    # 填充零动作
                    actions.append(np.zeros(5))
            actions_list.append(actions)

            # 生成目标
            targets = game.make_target(
                game_pos,
                self.num_unroll_steps,
                self.td_steps
            )
            targets_list.append(targets)

        return observations, actions_list, targets_list

    def __len__(self):
        """缓冲区中的游戏数"""
        return len(self.buffer)

    def save_to_disk(self, filepath: str):
        """保存缓冲区到磁盘"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"已保存 {len(self.buffer)} 局游戏到 {filepath}")

    def load_from_disk(self, filepath: str):
        """从磁盘加载缓冲区"""
        try:
            with open(filepath, 'rb') as f:
                games = pickle.load(f)
            self.buffer.extend(games)
            print(f"从 {filepath} 加载了 {len(games)} 局游戏")
        except FileNotFoundError:
            print(f"文件 {filepath} 不存在，跳过加载")


class GameRecorder:
    """游戏记录器：在游戏过程中收集数据"""

    def __init__(self):
        """初始化记录器"""
        self.current_game = Game()

    def new_game(self, my_identity: str):
        """开始新游戏

        参数：
            my_identity: 我的身份 ('A' or 'B')
        """
        self.current_game = Game()
        self.current_game.my_identity = my_identity

    def record_step(self,
                   observation: np.ndarray,
                   action: np.ndarray,
                   reward: float,
                   policy: Tuple[np.ndarray, np.ndarray],
                   value: float):
        """记录一步"""
        self.current_game.store_transition(
            observation, action, reward, policy, value
        )

    def end_game(self, winner: str, hit_count: int) -> Game:
        """结束游戏并返回记录

        参数：
            winner: 获胜者 ('A', 'B', 'SAME')
            hit_count: 总击球数

        返回：
            完整的游戏记录
        """
        self.current_game.winner = winner
        self.current_game.hit_count = hit_count
        return self.current_game


def compute_reward_from_step_info(step_info: Dict,
                                  player_targets: List[str],
                                  balls_before: Dict,
                                  balls_after: Dict) -> float:
    """根据环境返回的信息计算奖励

    参数：
        step_info: env.take_shot() 返回的信息
        player_targets: 当前玩家的目标球
        balls_before: 击球前的球状态
        balls_after: 击球后的球状态

    返回：
        reward: 标量奖励
    """
    reward = 0.0

    # 我方进球奖励
    my_pocketed = step_info.get('ME_INTO_POCKET', [])
    reward += len(my_pocketed) * 50

    # 对方进球惩罚
    enemy_pocketed = step_info.get('ENEMY_INTO_POCKET', [])
    reward -= len(enemy_pocketed) * 20

    # 白球进袋惩罚
    if step_info.get('WHITE_BALL_INTO_POCKET', False):
        reward -= 100

    # 黑8进袋（需要判断是否合法）
    if step_info.get('BLACK_BALL_INTO_POCKET', False):
        remaining_targets = [tid for tid in player_targets
                           if tid in balls_before and balls_before[tid].state.s != 4]
        if len(remaining_targets) == 0:
            reward += 200  # 合法胜利
        else:
            reward -= 200  # 非法失败

    # 犯规惩罚
    if step_info.get('FOUL_FIRST_HIT', False):
        reward -= 30
    if step_info.get('NO_POCKET_NO_RAIL', False):
        reward -= 30
    if step_info.get('NO_HIT', False):
        reward -= 30

    # 无事发生的基础奖励
    if reward == 0:
        reward = 5  # 小奖励，鼓励保持游戏进行

    return reward


def test_replay_buffer():
    """测试重放缓冲区"""
    print("=== 测试重放缓冲区 ===")

    # 创建缓冲区
    replay = ReplayBuffer(max_size=100, batch_size=4, num_unroll_steps=5)

    # 创建模拟游戏
    for game_idx in range(10):
        game = Game()
        game.my_identity = 'A'
        game.winner = random.choice(['A', 'B', 'SAME'])

        # 添加随机步
        for step in range(random.randint(10, 30)):
            obs = np.random.randn(83)
            action = np.random.randn(5)
            reward = random.uniform(-10, 10)
            policy_mu = np.random.randn(5)
            policy_sigma = np.abs(np.random.randn(5)) + 0.1
            value = random.uniform(-50, 50)

            game.store_transition(obs, action, reward, (policy_mu, policy_sigma), value)

        replay.save_game(game)

    print(f"缓冲区大小: {len(replay)} 局游戏")

    # 采样批次
    obs_batch, actions_batch, targets_batch = replay.sample_batch()

    print(f"采样批次大小: {len(obs_batch)}")
    print(f"观测形状: {obs_batch[0].shape}")
    print(f"动作序列长度: {len(actions_batch[0])}")
    print(f"目标值数量: {len(targets_batch[0]['value'])}")

    print("\n✓ 重放缓冲区测试通过！")


if __name__ == '__main__':
    test_replay_buffer()
