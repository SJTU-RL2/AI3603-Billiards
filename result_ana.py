import re
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics
import math


class GameLog:
    """单局游戏日志"""
    def __init__(self):
        self.game_number = None
        self.player_a_agent = None
        self.player_b_agent = None
        self.player_a_target = None
        self.shots = []  # [(shot_number, player, timestamp), ...]
        self.winner = None
        self.duration = None
        self.opponent_ball_pocketed = []  # 对方球入袋事件
        
    def get_agent_for_player(self, player):
        """根据Player获取对应的Agent名称"""
        if player == 'A':
            return self.player_a_agent
        elif player == 'B':
            return self.player_b_agent
        return None


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self):
        self.agent_a_name = None
        self.agent_b_name = None
        self.games = []
        
    def parse_log_file(self, file_path):
        """解析单个日志文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_game = None
        last_timestamp = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 提取时间戳
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                last_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                content = line[len(timestamp_str):].strip(' -')
            else:
                content = line
            
            # 解析Agent名称（全局信息）- 只在开始评估时解析，避免匹配到战绩统计行
            if '开始对战评估' in content or ('Agent A:' in content and 'Agent B:' in content and '累计战绩' not in content and '局结果' not in content):
                match = re.search(r'Agent A:\s*(\w+),\s*Agent B:\s*(\w+)', content)
                if match and not self.agent_a_name:  # 只设置一次
                    self.agent_a_name = match.group(1)
                    self.agent_b_name = match.group(2)
            
            # 新局开始
            if '局比赛开始' in content:
                match = re.search(r'第\s*(\d+)\s*局比赛开始', content)
                if match:
                    current_game = GameLog()
                    current_game.game_number = int(match.group(1))
            
            # 本局Player和Agent映射
            if current_game and '本局 Player A:' in content:
                match = re.search(r'Player A:\s*(\w+),\s*目标球型:\s*(\w+)', content)
                if match:
                    current_game.player_a_agent = match.group(1)
                    current_game.player_a_target = match.group(2)
                    # 推断Player B的Agent
                    if current_game.player_a_agent == self.agent_a_name:
                        current_game.player_b_agent = self.agent_b_name
                    else:
                        current_game.player_b_agent = self.agent_a_name
            
            # 击球记录
            if current_game and '次击球' in content:
                match = re.search(r'\[第(\d+)次击球\]\s*player:\s*([AB])', content)
                if match and last_timestamp:
                    shot_number = int(match.group(1))
                    player = match.group(2)
                    current_game.shots.append((shot_number, player, last_timestamp))
            
            # 对方球入袋
            if current_game and '对方球入袋' in content:
                current_game.opponent_ball_pocketed.append(last_timestamp)
            
            # 局结果 - 这里的Agent A/B指的是全局Agent
            if current_game and '局结果' in content:
                match = re.search(r'Agent ([AB])\s*获胜', content)
                if match:
                    winner_label = match.group(1)  # 'A' 或 'B'
                    # 全局的Agent A/B
                    if winner_label == 'A':
                        current_game.winner = self.agent_a_name
                    else:
                        current_game.winner = self.agent_b_name
            
            # 本局耗时
            if current_game and '本局耗时' in content:
                match = re.search(r'本局耗时:\s*([\d.]+)s', content)
                if match:
                    current_game.duration = float(match.group(1))
                    self.games.append(current_game)
                    current_game = None
    
    def analyze(self):
        """执行分析"""
        if not self.games:
            print("没有找到有效的游戏数据")
            return
        
        print("\n" + "="*80)
        print(f"日志分析报告")
        print("="*80)
        print(f"\n总局数: {len(self.games)}")
        print(f"Agent A: {self.agent_a_name}")
        print(f"Agent B: {self.agent_b_name}")
        
        # 1. 胜率统计
        self._analyze_win_rate()
        
        # 2. 平均决策时间
        self._analyze_decision_time()
        
        # 3. 获胜原因分析
        self._analyze_win_reason()
        
        # 4. 平均轮次
        self._analyze_average_rounds()
        
        # 5. 决策占比
        self._analyze_decision_ratio()
        
        # 6. 连续击球分析
        self._analyze_consecutive_shots()
        
        # 7. 开局表现
        self._analyze_first_move_advantage()
        
        # 8. 胜率置信区间
        self._analyze_confidence_interval()
        
        # 9. 决策时间分布
        self._analyze_decision_time_distribution()
    
    def _analyze_win_rate(self):
        """分析胜率"""
        print("\n" + "-"*80)
        print("1. 胜率统计")
        print("-"*80)
        
        wins = {self.agent_a_name: 0, self.agent_b_name: 0}
        for game in self.games:
            if game.winner in wins:
                wins[game.winner] += 1
        
        total = len(self.games)
        for agent, win_count in wins.items():
            win_rate = win_count / total * 100
            print(f"{agent}: {win_count}胜 / {total}局 = {win_rate:.2f}%")
    
    def _analyze_decision_time(self):
        """分析平均决策时间"""
        print("\n" + "-"*80)
        print("2. 平均决策时间")
        print("-"*80)
        
        agent_times = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            for i in range(1, len(game.shots)):
                prev_shot = game.shots[i-1]
                curr_shot = game.shots[i]
                
                # 计算决策时间（当前击球时间 - 上一次击球时间）
                time_diff = (curr_shot[2] - prev_shot[2]).total_seconds()
                
                # 获取当前击球的Agent
                agent = game.get_agent_for_player(curr_shot[1])
                if agent in agent_times:
                    agent_times[agent].append(time_diff)
        
        for agent, times in agent_times.items():
            if times:
                avg_time = statistics.mean(times)
                print(f"{agent}: 平均 {avg_time:.2f}秒 (共{len(times)}次决策)")
            else:
                print(f"{agent}: 无数据")
    
    def _analyze_win_reason(self):
        """分析获胜原因"""
        print("\n" + "-"*80)
        print("3. 获胜原因分析")
        print("-"*80)
        
        win_reasons = {
            self.agent_a_name: {'正常打入黑8': 0, '对方失误': 0},
            self.agent_b_name: {'正常打入黑8': 0, '对方失误': 0}
        }
        
        for game in self.games:
            if not game.shots or not game.winner:
                continue
            
            # 最后一次击球的Player
            last_shot_player = game.shots[-1][1]
            last_shot_agent = game.get_agent_for_player(last_shot_player)
            
            if last_shot_agent == game.winner:
                win_reasons[game.winner]['正常打入黑8'] += 1
            else:
                win_reasons[game.winner]['对方失误'] += 1
        
        for agent, reasons in win_reasons.items():
            total_wins = reasons['正常打入黑8'] + reasons['对方失误']
            if total_wins > 0:
                normal_rate = reasons['正常打入黑8'] / total_wins * 100
                mistake_rate = reasons['对方失误'] / total_wins * 100
                print(f"\n{agent} (共{total_wins}胜):")
                print(f"  正常打入黑8: {reasons['正常打入黑8']}次 ({normal_rate:.1f}%)")
                print(f"  对方失误: {reasons['对方失误']}次 ({mistake_rate:.1f}%)")
    
    def _analyze_average_rounds(self):
        """分析平均轮次"""
        print("\n" + "-"*80)
        print("4. 平均轮次统计")
        print("-"*80)
        
        total_shots = [len(game.shots) for game in self.games if game.shots]
        if total_shots:
            avg_shots = statistics.mean(total_shots)
            min_shots = min(total_shots)
            max_shots = max(total_shots)
            median_shots = statistics.median(total_shots)
            
            print(f"平均每局击球次数: {avg_shots:.2f}次")
            print(f"中位数: {median_shots:.0f}次")
            print(f"最少: {min_shots}次, 最多: {max_shots}次")
    
    def _analyze_decision_ratio(self):
        """分析决策占比"""
        print("\n" + "-"*80)
        print("5. 决策占比分析")
        print("-"*80)
        
        agent_shot_counts = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            shot_count = {self.agent_a_name: 0, self.agent_b_name: 0}
            
            for shot in game.shots:
                agent = game.get_agent_for_player(shot[1])
                if agent in shot_count:
                    shot_count[agent] += 1
            
            total = sum(shot_count.values())
            if total > 0:
                for agent in agent_shot_counts:
                    ratio = shot_count[agent] / total
                    agent_shot_counts[agent].append(ratio)
        
        for agent, ratios in agent_shot_counts.items():
            if ratios:
                avg_ratio = statistics.mean(ratios) * 100
                print(f"{agent}: 平均每局决策占比 {avg_ratio:.2f}%")
    
    def _analyze_consecutive_shots(self):
        """分析连续击球"""
        print("\n" + "-"*80)
        print("6. 连续击球分析")
        print("-"*80)
        
        agent_consecutive = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            if len(game.shots) < 2:
                continue
            
            current_agent = game.get_agent_for_player(game.shots[0][1])
            consecutive_count = 1
            
            for i in range(1, len(game.shots)):
                agent = game.get_agent_for_player(game.shots[i][1])
                
                if agent == current_agent:
                    consecutive_count += 1
                else:
                    # 记录连续击球次数
                    if current_agent in agent_consecutive and consecutive_count > 1:
                        agent_consecutive[current_agent].append(consecutive_count)
                    current_agent = agent
                    consecutive_count = 1
            
            # 记录最后一段连续击球
            if current_agent in agent_consecutive and consecutive_count > 1:
                agent_consecutive[current_agent].append(consecutive_count)
        
        for agent, consecutives in agent_consecutive.items():
            if consecutives:
                avg_consecutive = statistics.mean(consecutives)
                max_consecutive = max(consecutives)
                print(f"{agent}: 平均连续击球 {avg_consecutive:.2f}次, 最高 {max_consecutive}次 (共{len(consecutives)}次连击)")
            else:
                print(f"{agent}: 无连续击球记录")
    
    def _analyze_first_move_advantage(self):
        """分析开局优势"""
        print("\n" + "-"*80)
        print("7. 开局表现分析")
        print("-"*80)
        
        first_move_wins = {self.agent_a_name: 0, self.agent_b_name: 0}
        first_move_total = {self.agent_a_name: 0, self.agent_b_name: 0}
        
        for game in self.games:
            if not game.shots or not game.winner:
                continue
            
            # 第一次击球的Agent
            first_agent = game.get_agent_for_player(game.shots[0][1])
            
            if first_agent in first_move_total:
                first_move_total[first_agent] += 1
                if first_agent == game.winner:
                    first_move_wins[first_agent] += 1
        
        for agent in [self.agent_a_name, self.agent_b_name]:
            total = first_move_total[agent]
            wins = first_move_wins[agent]
            if total > 0:
                win_rate = wins / total * 100
                print(f"{agent} 先手: {wins}胜/{total}局 = {win_rate:.2f}%")
    
    def _analyze_confidence_interval(self):
        """分析胜率置信区间（95%置信区间）"""
        print("\n" + "-"*80)
        print("8. 胜率置信区间 (95%)")
        print("-"*80)
        
        wins = {self.agent_a_name: 0, self.agent_b_name: 0}
        for game in self.games:
            if game.winner in wins:
                wins[game.winner] += 1
        
        n = len(self.games)
        z = 1.96  # 95%置信区间的z值
        
        for agent, win_count in wins.items():
            p = win_count / n
            se = math.sqrt(p * (1 - p) / n)
            ci_lower = max(0, p - z * se) * 100
            ci_upper = min(1, p + z * se) * 100
            
            print(f"{agent}: {p*100:.2f}% (95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%])")
    
    def _analyze_decision_time_distribution(self):
        """分析决策时间分布"""
        print("\n" + "-"*80)
        print("9. 决策时间分布")
        print("-"*80)
        
        agent_times = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            for i in range(1, len(game.shots)):
                prev_shot = game.shots[i-1]
                curr_shot = game.shots[i]
                time_diff = (curr_shot[2] - prev_shot[2]).total_seconds()
                agent = game.get_agent_for_player(curr_shot[1])
                if agent in agent_times:
                    agent_times[agent].append(time_diff)
        
        for agent, times in agent_times.items():
            if times:
                avg = statistics.mean(times)
                median = statistics.median(times)
                stdev = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)
                
                print(f"\n{agent}:")
                print(f"  平均值: {avg:.2f}秒")
                print(f"  中位数: {median:.2f}秒")
                print(f"  标准差: {stdev:.2f}秒")
                print(f"  范围: [{min_time:.2f}秒, {max_time:.2f}秒]")


def main(log_path):
    """主函数"""
    import sys
    from io import StringIO
    
    analyzer = LogAnalyzer()
    
    if os.path.isfile(log_path):
        print(f"正在分析文件: {log_path}")
        analyzer.parse_log_file(log_path)
        output_path = log_path.rsplit('.', 1)[0] + "_analysis.txt"
    elif os.path.isdir(log_path):
        log_files = list(Path(log_path).glob("*.log"))
        if not log_files:
            print(f"在 {log_path} 中没有找到 .log 文件")
            return
        
        print(f"找到 {len(log_files)} 个日志文件")
        for log_file in log_files:
            print(f"  - {log_file.name}")
            analyzer.parse_log_file(str(log_file))
        output_path = os.path.join(log_path, "combined_analysis.txt")
    else:
        print(f"错误: {log_path} 不是有效的文件或文件夹")
        return
    
    # 捕获标准输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    analyzer.analyze()
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)
    
    # 获取输出内容
    output_content = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # 打印到控制台
    print(output_content)
    
    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"\n分析结果已保存到: {output_path}")


if __name__ == "__main__":
    # 直接在这里修改要分析的路径
    log_path = "logs_"  # 可以是文件路径或文件夹路径
    # log_path = "logs/evaluate_20251212_134237.log"  # 单个文件示例
    
    main(log_path)
