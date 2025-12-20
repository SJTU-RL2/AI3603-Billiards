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
import time
import logging
from datetime import datetime
import os

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 配置评估日志 - 使用独立的logger避免被poolenv的basicConfig覆盖
eval_log_filename = "logs/evaluate.log"
eval_logger = logging.getLogger('evaluate')
eval_logger.setLevel(logging.INFO)
eval_logger.propagate = False  # 不传播到root logger

# 创建文件和控制台处理器（续写模式）
file_handler = logging.FileHandler(eval_log_filename, mode='a', encoding='utf-8')
stream_handler = logging.StreamHandler()

# 设置格式
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# 添加处理器
eval_logger.addHandler(file_handler)
eval_logger.addHandler(stream_handler)

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=False, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 300  # 对战局数 自己测试时可以修改 扩充为120局为了减少随机带来的扰动

agent_a, agent_b = BasicAgent(), NewAgent()

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

# 统计数据
game_times = []  # 每局用时
overtime_games = 0  # 超时对局数（>3分钟）
break_stats = {
    'AGENT_A': {'break_count': 0, 'win_count': 0},  # A开球的统计
    'AGENT_B': {'break_count': 0, 'win_count': 0}   # B开球的统计
}
game_details = []  # 每局详细信息

eval_logger.info("=" * 80)
eval_logger.info(f"评估开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
eval_logger.info(f"Agent A: {agent_a.__class__.__name__}")
eval_logger.info(f"Agent B: {agent_b.__class__.__name__}")
eval_logger.info(f"总对战局数: {n_games}")
eval_logger.info("=" * 80)
eval_logger.info("")

for i in range(n_games): 
    game_start_time = time.time()
    
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    eval_logger.info(f"{'=' * 40}")
    eval_logger.info(f"第 {i+1}/{n_games} 局比赛")
    
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    
    # 记录开球方
    breaker = 'AGENT_A' if i % 2 == 0 else 'AGENT_B'
    break_stats[breaker]['break_count'] += 1
    
    print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
    eval_logger.info(f"Player A 使用: {player_class}")
    eval_logger.info(f"Player A 目标球型: {ball_type}")
    eval_logger.info(f"开球方: {breaker}")
    
    winner = None
    win_reason = "未知"
    last_step_info = {}  # 保存最后一次step_info用于判断获胜原因
    
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        step_info = env.take_shot(action)
        last_step_info = step_info  # 保存每次的step_info
        
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
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            game_times.append(game_duration)
            
            # 统计超时对局（超过3分钟）
            if game_duration > 180:
                overtime_games += 1
            
            # 确定获胜原因 - 使用保存的last_step_info
            # 注意：必须使用 == True 来严格判断，因为 get() 可能返回 None
            if info['winner'] == 'SAME':
                win_reason = "平局（超过最大回合数）"
            elif (last_step_info.get('WHITE_BALL_INTO_POCKET') == True and 
                  last_step_info.get('BLACK_BALL_INTO_POCKET') == True):
                win_reason = "对手白球与黑8同时进袋"
            elif last_step_info.get('BLACK_BALL_INTO_POCKET') == True:
                # 检查是否合法打进黑8（通过判断winner是否是打球方）
                # 如果黑8进袋且当前winner不是打球方，说明是非法打进
                current_player = player  # 最后打球的是谁（循环中的player）
                # 注意：done时winner已经确定，如果winner不是当前打球方，说明是犯规
                if info['winner'] == current_player:
                    win_reason = "合法打进黑8     "
                else:
                    win_reason = "对手非法打进黑8   "
            elif last_step_info.get('WHITE_BALL_INTO_POCKET') == True:
                win_reason = "对手白球进袋犯规（关键时刻）"
            elif last_step_info.get('NO_HIT') == True:
                win_reason = "对手白球未接触任何球犯规"
            elif last_step_info.get('FOUL_FIRST_HIT') == True:
                win_reason = "对手首次接触违规球犯规"
            elif last_step_info.get('NO_POCKET_NO_RAIL') == True:
                win_reason = "对手无进球且未碰库犯规"
            else:
                win_reason = "对手犯规或其他原因"
            
            # 统计结果（player A/B 转换为 agent A/B） 
            if info['winner'] == 'SAME':
                results['SAME'] += 1
                winner = 'SAME'
            elif info['winner'] == 'A':
                actual_winner = ['AGENT_A', 'AGENT_B'][i % 2]
                results[actual_winner + '_WIN'] += 1
                winner = actual_winner
                # 如果是开球方获胜，记录开球胜率
                if breaker == actual_winner:
                    break_stats[breaker]['win_count'] += 1
            else:
                actual_winner = ['AGENT_A', 'AGENT_B'][(i+1) % 2]
                results[actual_winner + '_WIN'] += 1
                winner = actual_winner
                # 如果是开球方获胜，记录开球胜率
                if breaker == actual_winner:
                    break_stats[breaker]['win_count'] += 1
            
            # 记录详细信息
            game_details.append({
                'game_num': i + 1,
                'breaker': breaker,
                'winner': winner,
                'win_reason': win_reason,
                'duration': game_duration,
                'hit_count': info['hit_count']
            })
            
            eval_logger.info(f"本局胜者: {winner}")
            eval_logger.info(f"获胜原因: {win_reason}")
            eval_logger.info(f"总击球次数: {info['hit_count']}")
            eval_logger.info(f"本局用时: {game_duration:.2f}秒")
            eval_logger.info("")
            
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n最终结果：", results)

# ============ 详细统计报告 ============
eval_logger.info("=" * 80)
eval_logger.info("详细统计报告")
eval_logger.info("=" * 80)
eval_logger.info("")

# 1. 基本胜负统计
eval_logger.info("【基本胜负统计】")
eval_logger.info(f"Agent A ({agent_a.__class__.__name__}) 胜: {results['AGENT_A_WIN']}局")
eval_logger.info(f"Agent B ({agent_b.__class__.__name__}) 胜: {results['AGENT_B_WIN']}局")
eval_logger.info(f"平局: {results['SAME']}局")
eval_logger.info(f"Agent A 得分: {results['AGENT_A_SCORE']:.1f}")
eval_logger.info(f"Agent B 得分: {results['AGENT_B_SCORE']:.1f}")
if n_games > 0:
    eval_logger.info(f"Agent A 胜率: {results['AGENT_A_WIN']/n_games*100:.1f}%")
    eval_logger.info(f"Agent B 胜率: {results['AGENT_B_WIN']/n_games*100:.1f}%")
eval_logger.info("")

# 2. 用时统计
if game_times:
    avg_time = sum(game_times) / len(game_times)
    max_time = max(game_times)
    min_time = min(game_times)
    total_time = sum(game_times)
    
    eval_logger.info("【用时统计】")
    eval_logger.info(f"总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    eval_logger.info(f"平均每局用时: {avg_time:.2f}秒")
    eval_logger.info(f"最长一局用时: {max_time:.2f}秒")
    eval_logger.info(f"最短一局用时: {min_time:.2f}秒")
    eval_logger.info(f"超时对局数（>3分钟）: {overtime_games}局 ({overtime_games/len(game_times)*100:.1f}%)")
    eval_logger.info("")

# 3. 开球胜率统计
eval_logger.info("【开球胜率统计】")
for agent_name, stats in break_stats.items():
    if stats['break_count'] > 0:
        break_win_rate = stats['win_count'] / stats['break_count'] * 100
        eval_logger.info(f"{agent_name}: 开球{stats['break_count']}局, 获胜{stats['win_count']}局, 胜率{break_win_rate:.1f}%")
    else:
        eval_logger.info(f"{agent_name}: 未开球")
eval_logger.info("")

# 4. 每局详细信息
eval_logger.info("【每局详细信息】")
eval_logger.info(f"{'局号':<6} {'开球方':<12} {'胜者':<12} {'获胜原因':<40} {'用时(秒)':<10} {'击球数':<8}")
eval_logger.info("-" * 100)
for detail in game_details:
    eval_logger.info(f"{detail['game_num']:<6} {detail['breaker']:<12} {detail['winner']:<12} "
                    f"{detail['win_reason']:<40} {detail['duration']:<10.2f} {detail['hit_count']:<8}")
eval_logger.info("")

# 5. AGENT_A获胜原因统计
eval_logger.info("【AGENT_A获胜原因分析】")
agent_a_win_reason_count = {}
for detail in game_details:
    if detail['winner'] == 'AGENT_A':
        reason = detail['win_reason']
        agent_a_win_reason_count[reason] = agent_a_win_reason_count.get(reason, 0) + 1

if agent_a_win_reason_count:
    for reason, count in sorted(agent_a_win_reason_count.items(), key=lambda x: x[1], reverse=True):
        eval_logger.info(f"{reason}: {count}次 ({count/results['AGENT_A_WIN']*100:.1f}%)")
else:
    eval_logger.info("AGENT_A未获胜")
eval_logger.info("")

eval_logger.info("=" * 80)
eval_logger.info(f"评估结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
eval_logger.info(f"日志文件: {eval_log_filename}")
eval_logger.info("=" * 80)