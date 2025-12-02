## SAC Agent (`NewAgent`)

- `agent.NewAgent` 现实现为基于 PyTorch 的 Soft Actor-Critic (SAC) 策略网络，推理阶段默认加载 `checkpoints/sac_agent.pth`（若不存在则使用随机初始化权重）。
- 训练阶段可将 `NewAgent(training=True)` 并为其提供环境交互数据，使用 `store_transition(...)`、`update_parameters()` 和 `save_checkpoint()` 完成离线或在线训练。
- 训练样例流程：
  1. `env = PoolEnv()`，对手可选 `BasicAgent()`；
  2. 在智能体轮到出杆时调用 `encode_observation()` 获得状态向量；
  3. 执行动作后用 `PoolEnv.take_shot` 的返回值通过 `compute_dense_reward()` 计算当前奖励；
  4. 调用 `store_transition(state, action, reward, next_state, done)` 写入经验池，并循环 `update_parameters()` 更新网络；
  5. 定期 `save_checkpoint()`，以便在 `evaluate.py` 中直接加载。
- 依赖：需安装 `torch`，已在 `requirements.txt` 中声明。

## 训练脚本 (`train/train.py`)

运行示例：

```bash
python train/train.py --episodes 50 --opponent basic --checkpoint checkpoints/sac_agent.pth
```

- `--opponent {basic, random}`：选择与 SAC 对战的对手，默认随机策略，`basic` 会调用 `BasicAgent`。
- `--control-player {A,B}`：指定由 SAC 控制的一方，默认 `A`。
- `--target-cycle`：如 `solid,stripe`，在多局训练中循环分配球型。
- `--env-noise`：开启后，环境击球参数会加入高斯噪声以模拟真实误差。
- `--learning-starts / --updates-per-step`：控制何时开始参数更新以及每次回合的梯度步数。
- `--checkpoint`、`--save-every`、`--log-dir`：分别控制模型保存路径、保存频率和指标输出位置（CSV 文件）。

脚本会自动记录训练指标至 `logs/training_metrics.csv`，并在训练过程中滚动保存 checkpoint，方便随时恢复或用于对战评估。