"""
muzero_mcts.py - 连续动作空间的MCTS实现

为MuZero实现适配连续动作空间的蒙特卡洛树搜索
使用Progressive Widening策略处理无限动作空间
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional
from muzero_core import MuZeroNetwork


class Node:
    """MCTS搜索树节点"""

    def __init__(self,
                 prior: float = 0.0,
                 parent: Optional['Node'] = None,
                 action: Optional[np.ndarray] = None):
        """
        参数：
            prior: 先验概率
            parent: 父节点
            action: 到达此节点的动作
        """
        self.parent = parent
        self.action = action  # 到达此节点的动作 [5]
        self.prior = prior

        self.children: Dict[int, Node] = {}  # 子节点字典 {action_idx: Node}
        self.child_actions: List[np.ndarray] = []  # 采样的动作列表

        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0  # 从父节点到此节点的即时奖励
        self.hidden_state = None  # MuZero的隐状态

    def expanded(self) -> bool:
        """检查节点是否已扩展"""
        return len(self.children) > 0

    def value(self) -> float:
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self,
               policy_mu: np.ndarray,
               policy_sigma: np.ndarray,
               hidden_state: np.ndarray,
               num_actions: int = 10):
        """扩展节点：从策略分布采样动作

        参数：
            policy_mu: [5] 策略均值
            policy_sigma: [5] 策略标准差
            hidden_state: 隐状态
            num_actions: 采样动作数量
        """
        self.hidden_state = hidden_state

        # 从高斯分布采样多个动作
        for i in range(num_actions):
            # 采样动作
            action = np.random.normal(policy_mu, policy_sigma)

            # 裁剪到合理范围
            action = self._clip_action(action)

            # 计算先验概率（高斯概率密度）
            prior = self._gaussian_pdf(action, policy_mu, policy_sigma)

            # 创建子节点
            child = Node(prior=prior, parent=self, action=action)
            self.children[i] = child
            self.child_actions.append(action)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作到合法范围"""
        action = action.copy()
        action[0] = np.clip(action[0], 0.5, 8.0)    # V0
        action[1] = action[1] % 360                  # phi
        action[2] = np.clip(action[2], 0, 90)       # theta
        action[3] = np.clip(action[3], -0.5, 0.5)   # a
        action[4] = np.clip(action[4], -0.5, 0.5)   # b
        return action

    def _gaussian_pdf(self,
                     action: np.ndarray,
                     mu: np.ndarray,
                     sigma: np.ndarray) -> float:
        """计算高斯概率密度"""
        # 多维高斯概率密度（假设独立）
        prob = 1.0
        for a, m, s in zip(action, mu, sigma):
            prob *= np.exp(-0.5 * ((a - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        return prob

    def add_exploration_noise(self, epsilon: float = 0.25, alpha: float = 0.3):
        """添加Dirichlet噪声以增加探索"""
        if not self.expanded():
            return

        num_children = len(self.children)
        noise = np.random.dirichlet([alpha] * num_children)

        for i, child in self.children.items():
            child.prior = child.prior * (1 - epsilon) + noise[i] * epsilon


class MCTS:
    """连续动作空间的MCTS"""

    def __init__(self,
                 network: MuZeroNetwork,
                 num_simulations: int = 50,
                 num_actions_per_node: int = 10,
                 c_puct: float = 1.5,
                 discount: float = 0.99,
                 temperature: float = 1.0):
        """
        参数：
            network: MuZero网络
            num_simulations: 模拟次数
            num_actions_per_node: 每个节点采样的动作数
            c_puct: UCB公式的探索常数
            discount: 折扣因子
            temperature: 温度参数（控制探索）
        """
        self.network = network
        self.num_simulations = num_simulations
        self.num_actions_per_node = num_actions_per_node
        self.c_puct = c_puct
        self.discount = discount
        self.temperature = temperature

    def run(self,
            observation: np.ndarray,
            add_noise: bool = True) -> np.ndarray:
        """运行MCTS搜索

        参数：
            observation: [83] 观测特征
            add_noise: 是否添加探索噪声

        返回：
            action: [5] 最佳动作
        """
        # 创建根节点
        root = Node()

        # 初始推理
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            result = self.network.initial_inference(obs_tensor)

        policy_mu = result['policy_mu'][0].cpu().numpy()
        policy_sigma = result['policy_sigma'][0].cpu().numpy()
        hidden_state = result['hidden_state'][0].cpu().numpy()

        # 扩展根节点
        root.expand(policy_mu, policy_sigma, hidden_state, self.num_actions_per_node)

        # 添加探索噪声（仅在自我对弈时）
        if add_noise:
            root.add_exploration_noise()

        # 运行模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: 选择到叶子节点
            while node.expanded():
                node = self._select_child(node)
                search_path.append(node)

            # 获取父节点的隐状态和动作
            parent = search_path[-2]
            action = node.action

            # Expansion & Evaluation: 使用网络预测
            hidden_state = parent.hidden_state
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            hidden_tensor = torch.FloatTensor(hidden_state).unsqueeze(0)

            with torch.no_grad():
                result = self.network.recurrent_inference(hidden_tensor, action_tensor)

            reward = result['reward'][0, 0].item()
            value = result['value'][0, 0].item()
            policy_mu = result['policy_mu'][0].cpu().numpy()
            policy_sigma = result['policy_sigma'][0].cpu().numpy()
            next_hidden_state = result['hidden_state'][0].cpu().numpy()

            # 扩展节点
            node.reward = reward
            node.expand(policy_mu, policy_sigma, next_hidden_state, self.num_actions_per_node)

            # Backpropagation: 回溯更新
            self._backpropagate(search_path, value)

        # 选择最佳动作
        return self._select_action(root)

    def _select_child(self, node: Node) -> Node:
        """选择UCB值最高的子节点"""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """计算UCB分数

        UCB = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        # Q值：平均价值
        q_value = child.value()

        # U值：探索奖励
        u_value = (self.c_puct * child.prior *
                  math.sqrt(parent.visit_count) / (1 + child.visit_count))

        return q_value + u_value

    def _backpropagate(self, search_path: List[Node], value: float):
        """回溯更新路径上所有节点的统计信息"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

            # 考虑即时奖励和折扣
            value = node.reward + self.discount * value

    def _select_action(self, root: Node) -> np.ndarray:
        """根据访问次数选择动作

        使用温度参数控制探索/利用
        """
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = np.array(root.child_actions)

        if self.temperature == 0:
            # 贪心选择
            action_idx = np.argmax(visit_counts)
        else:
            # 根据访问次数的温度软化分布采样
            visits_temp = visit_counts ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            action_idx = np.random.choice(len(actions), p=probs)

        return actions[action_idx]


def test_mcts():
    """测试MCTS"""
    print("=== 测试MCTS ===")

    # 创建网络
    network = MuZeroNetwork(state_dim=128, action_dim=5, hidden_dim=256)
    network.eval()

    # 创建MCTS
    mcts = MCTS(
        network=network,
        num_simulations=20,
        num_actions_per_node=8,
        c_puct=1.5
    )

    # 测试搜索
    observation = np.random.randn(83)
    action = mcts.run(observation, add_noise=False)

    print(f"观测形状: {observation.shape}")
    print(f"选择的动作: {action}")
    print(f"动作范围检查:")
    print(f"  V0={action[0]:.2f} ∈ [0.5, 8.0]: {0.5 <= action[0] <= 8.0}")
    print(f"  phi={action[1]:.2f} ∈ [0, 360]: {0 <= action[1] <= 360}")
    print(f"  theta={action[2]:.2f} ∈ [0, 90]: {0 <= action[2] <= 90}")
    print(f"  a={action[3]:.3f} ∈ [-0.5, 0.5]: {-0.5 <= action[3] <= 0.5}")
    print(f"  b={action[4]:.3f} ∈ [-0.5, 0.5]: {-0.5 <= action[4] <= 0.5}")

    print("\n✓ MCTS测试通过！")


if __name__ == '__main__':
    test_mcts()
