"""
test_muzero.py - MuZero组件测试脚本

快速测试所有MuZero组件是否正常工作
"""

import sys
import traceback


def test_imports():
    """测试依赖导入"""
    print("=" * 60)
    print("测试1: 检查依赖导入")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  设备名: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch未安装，请运行: pip install torch")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy未安装")
        return False

    try:
        import pooltool as pt
        print(f"✓ Pooltool")
    except ImportError:
        print("✗ Pooltool未安装，请运行: pip install pooltool-billiards")
        return False

    print("\n所有依赖已安装！\n")
    return True


def test_network():
    """测试核心网络"""
    print("=" * 60)
    print("测试2: MuZero核心网络")
    print("=" * 60)

    try:
        from muzero_core import MuZeroNetwork
        import torch

        network = MuZeroNetwork(state_dim=128, action_dim=5, hidden_dim=256)
        print(f"✓ 网络创建成功")
        print(f"  参数量: {sum(p.numel() for p in network.parameters()):,}")

        # 测试前向传播
        obs = torch.randn(4, 83)
        result = network.initial_inference(obs)
        print(f"✓ 初始推理成功")
        print(f"  隐状态: {result['hidden_state'].shape}")
        print(f"  策略: {result['policy_mu'].shape}")
        print(f"  价值: {result['value'].shape}")

        action = torch.randn(4, 5)
        result = network.recurrent_inference(result['hidden_state'], action)
        print(f"✓ 递归推理成功")
        print(f"  奖励: {result['reward'].shape}")

        print("\n网络测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 网络测试失败: {e}")
        traceback.print_exc()
        return False


def test_mcts():
    """测试MCTS"""
    print("=" * 60)
    print("测试3: MCTS搜索")
    print("=" * 60)

    try:
        from muzero_core import MuZeroNetwork
        from muzero_mcts import MCTS
        import numpy as np

        network = MuZeroNetwork(state_dim=128, action_dim=5, hidden_dim=256)
        network.eval()

        mcts = MCTS(
            network=network,
            num_simulations=10,  # 少量模拟用于快速测试
            num_actions_per_node=5
        )

        observation = np.random.randn(83)
        action = mcts.run(observation, add_noise=False)

        print(f"✓ MCTS搜索成功")
        print(f"  动作: V0={action[0]:.2f}, phi={action[1]:.2f}, "
              f"theta={action[2]:.2f}, a={action[3]:.3f}, b={action[4]:.3f}")
        print(f"  动作范围检查:")
        print(f"    V0 ∈ [0.5, 8.0]: {0.5 <= action[0] <= 8.0}")
        print(f"    phi ∈ [0, 360]: {0 <= action[1] <= 360}")
        print(f"    theta ∈ [0, 90]: {0 <= action[2] <= 90}")

        print("\nMCTS测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ MCTS测试失败: {e}")
        traceback.print_exc()
        return False


def test_replay_buffer():
    """测试重放缓冲区"""
    print("=" * 60)
    print("测试4: 重放缓冲区")
    print("=" * 60)

    try:
        from muzero_replay import ReplayBuffer, Game
        import numpy as np

        replay = ReplayBuffer(max_size=100, batch_size=4)

        # 创建模拟游戏
        game = Game()
        game.my_identity = 'A'
        game.winner = 'A'

        for _ in range(10):
            obs = np.random.randn(83)
            action = np.random.randn(5)
            reward = np.random.randn()
            policy = (np.random.randn(5), np.abs(np.random.randn(5)) + 0.1)
            value = np.random.randn()
            game.store_transition(obs, action, reward, policy, value)

        replay.save_game(game)
        print(f"✓ 游戏保存成功")
        print(f"  缓冲区大小: {len(replay)} 局")

        obs_batch, actions_batch, targets_batch = replay.sample_batch()
        print(f"✓ 批次采样成功")
        print(f"  批量大小: {len(obs_batch)}")

        print("\n重放缓冲区测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 重放缓冲区测试失败: {e}")
        traceback.print_exc()
        return False


def test_trainer():
    """测试训练器"""
    print("=" * 60)
    print("测试5: 训练器")
    print("=" * 60)

    try:
        from muzero_core import MuZeroNetwork
        from muzero_trainer import MuZeroTrainer
        import numpy as np

        network = MuZeroNetwork(state_dim=128, action_dim=5, hidden_dim=256)
        trainer = MuZeroTrainer(network=network, device='cpu')

        # 创建模拟批次
        batch_size = 4
        num_unroll_steps = 3

        observations = [np.random.randn(83) for _ in range(batch_size)]
        actions_list = [[np.random.randn(5) for _ in range(num_unroll_steps + 1)]
                       for _ in range(batch_size)]
        targets_list = []
        for _ in range(batch_size):
            targets = {
                'value': [np.random.randn() for _ in range(num_unroll_steps + 1)],
                'reward': [np.random.randn() for _ in range(num_unroll_steps + 1)],
                'policy_mu': [np.random.randn(5) for _ in range(num_unroll_steps + 1)],
                'policy_sigma': [np.abs(np.random.randn(5)) + 0.1
                                for _ in range(num_unroll_steps + 1)]
            }
            targets_list.append(targets)

        losses = trainer.train_batch(observations, actions_list, targets_list)

        print(f"✓ 训练成功")
        print(f"  损失:")
        for key, value in losses.items():
            print(f"    {key}: {value:.4f}")

        print("\n训练器测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ 训练器测试失败: {e}")
        traceback.print_exc()
        return False


def test_agent():
    """测试MuZeroAgent"""
    print("=" * 60)
    print("测试6: MuZeroAgent")
    print("=" * 60)

    try:
        from agent import MuZeroAgent
        import pooltool as pt

        # 创建agent（无检查点，使用随机初始化）
        agent = MuZeroAgent(
            checkpoint_path=None,
            num_simulations=10,
            temperature=0.0
        )

        # 创建模拟环境
        table = pt.Table.default()
        balls = pt.get_rack(pt.GameType.EIGHTBALL, table)
        my_targets = ['1', '2', '3', '4', '5', '6', '7']

        # 决策
        action = agent.decision(balls, my_targets, table)

        print(f"✓ Agent决策成功")
        print(f"  动作: {action}")
        print(f"  类型: {type(action)}")
        print(f"  键: {list(action.keys())}")

        print("\nMuZeroAgent测试通过！\n")
        return True

    except Exception as e:
        print(f"✗ Agent测试失败: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "MuZero组件测试套件" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    tests = [
        ("依赖导入", test_imports),
        ("核心网络", test_network),
        ("MCTS搜索", test_mcts),
        ("重放缓冲区", test_replay_buffer),
        ("训练器", test_trainer),
        ("MuZeroAgent", test_agent)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n测试 '{name}' 遇到异常: {e}\n")
            traceback.print_exc()
            results.append((name, False))

    # 汇总结果
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！MuZero实现就绪。")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 开始训练: python train_muzero.py --num_epochs 20")
        print("  2. 查看文档: MUZERO_README.md")
        print()
        return True
    else:
        print("\n" + "=" * 60)
        print("⚠️  部分测试失败，请检查错误信息")
        print("=" * 60)
        print()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
