"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
3. 运行脚本查看结果
"""

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=False, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 120  # 对战局数 自己测试时可以修改 扩充为120局为了减少随机带来的扰动

agent_a, agent_b = BasicAgent(), NewAgent()

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

for i in range(n_games): 
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            # poolenv中已有打印，无需再输出
            # if step_info.get('FOUL_FIRST_HIT'):
            #     print("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            # if step_info.get('NO_POCKET_NO_RAIL'):
            #     print("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            # if step_info.get('NO_HIT'):
            #     print("本杆判罚：白球未接触任何球，直接交换球权。")
            # if step_info.get('ME_INTO_POCKET'):
            #     print(f"我方球入袋：{step_info['ME_INTO_POCKET']}")
            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        if done:
            # 统计结果（player A/B 转换为 agent A/B） 
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n最终结果：", results)