"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配
- 记录所有打印内容到日志文件
- 记录每局和总体耗时

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
3. 运行脚本查看结果，日志将保存在 logs/ 目录下
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent


def setup_logger(log_name='evaluate'):
    """
    设置日志记录器，同时输出到控制台和文件
    
    参数：
        log_name: 日志文件名前缀
    
    返回：
        logger: 配置好的日志记录器
        log_file: 日志文件路径
    """
    # 创建 logs 目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{log_name}_{timestamp}.log'
    
    # 创建日志记录器
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已存在的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 防止日志重复输出
    logger.propagate = False
    
    return logger, log_file


# ============ 日志设置 ============
logger, log_file = setup_logger('evaluate')

logger.info(f"日志文件已保存到: {log_file}")
logger.info("=" * 80)

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=False, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 120  # 对战局数 自己测试时可以修改 扩充为120局为了减少随机带来的扰动

agent_a, agent_b = BasicAgent(), NewAgent()

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

logger.info(f"开始对战评估，共进行 {n_games} 局")
logger.info(f"Agent A: {agent_a.__class__.__name__}, Agent B: {agent_b.__class__.__name__}")
logger.info("=" * 80)

# 记录总体开始时间
overall_start_time = time.time()

for i in range(n_games):
    game_start_time = time.time()
    
    logger.info("")
    logger.info(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    logger.info(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
    
    while True:
        player = env.get_curr_player()
        logger.info(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            if step_info.get('ENEMY_INTO_POCKET'):
                logger.info(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        
        if done:
            # 统计结果（player A/B 转换为 agent A/B）
            if info['winner'] == 'SAME':
                results['SAME'] += 1
                logger.info(f"第 {i} 局结果: 平局")
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                winner_name = ['Agent A', 'Agent B'][i % 2]
                logger.info(f"第 {i} 局结果: {winner_name} 获胜")
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                winner_name = ['Agent B', 'Agent A'][i % 2]
                logger.info(f"第 {i} 局结果: {winner_name} 获胜")
            
            # 统计时间
            game_elapsed_time = time.time() - game_start_time
            total_elapsed_time = time.time() - overall_start_time
            avg_time_per_game = total_elapsed_time / (i + 1)
            
            logger.info(f"累计战绩 - Agent A: {results['AGENT_A_WIN']}胜, Agent B: {results['AGENT_B_WIN']}胜, 平局: {results['SAME']}")
            logger.info(f"本局耗时: {game_elapsed_time:.2f}s | 总耗时: {total_elapsed_time:.2f}s | 平均耗时/局: {avg_time_per_game:.2f}s")
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

# 统计总体耗时
overall_elapsed_time = time.time() - overall_start_time

# 输出最终结果
logger.info("")
logger.info("=" * 80)
logger.info("最终评估结果")
logger.info("=" * 80)
logger.info(f"Agent A 胜利次数: {results['AGENT_A_WIN']}")
logger.info(f"Agent B 胜利次数: {results['AGENT_B_WIN']}")
logger.info(f"平局次数: {results['SAME']}")
logger.info(f"Agent A 总得分: {results['AGENT_A_SCORE']:.1f}")
logger.info(f"Agent B 总得分: {results['AGENT_B_SCORE']:.1f}")
logger.info(f"Agent A 胜率: {results['AGENT_A_SCORE'] / n_games * 100:.2f}%")
logger.info(f"Agent B 胜率: {results['AGENT_B_SCORE'] / n_games * 100:.2f}%")
logger.info("=" * 80)
logger.info("耗时统计")
logger.info("=" * 80)
logger.info(f"总耗时: {overall_elapsed_time:.2f}s ({overall_elapsed_time / 60:.2f} 分钟)")
logger.info(f"平均耗时/局: {overall_elapsed_time / n_games:.2f}s")
logger.info("=" * 80)
logger.info(f"日志文件已保存到: {log_file}")
