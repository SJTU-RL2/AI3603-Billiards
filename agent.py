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
import signal
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        在支持 SIGALRM 的平台（通常为 Unix/Linux）上，使用 signal.SIGALRM 实现硬超时。
        在 Windows 或不支持 SIGALRM/alarm 的环境、或非主线程中，自动降级为直接调用
        pt.simulate（不做硬超时），以避免因 SIGALRM 不存在导致评测直接失败。
    """
    # Windows 没有 SIGALRM；另外 signal.* 通常只能在主线程使用。
    if not (hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")):
        try:
            pt.simulate(shot, inplace=True)
            return True
        except Exception as e:
            logger.warning(f"物理模拟失败（无超时保护降级路径）: {e}")
            return False

    try:
        # 设置超时信号处理器（非主线程可能失败，此时降级）
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)  # 设置超时时间
        except Exception as e:
            logger.debug(f"无法启用SIGALRM超时保护，降级为直接模拟: {e}")
            pt.simulate(shot, inplace=True)
            return True

        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        try:
            signal.alarm(0)  # 取消超时
        except Exception:
            pass
        logger.warning(f"物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception:
        try:
            signal.alarm(0)  # 取消超时
        except Exception:
            pass
        raise
    finally:
        # 恢复原处理器（若 signal 在非主线程不可用，这里也要容错）
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
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
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 5000  # 白球+黑8同时进袋，一票否决
    elif cue_pocketed:
        score -= 1000  # 白球进袋（提高惩罚，与Agent评估一致）
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 5000  # 清台前误打黑8，一票否决
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 150  # 提高未碰库犯规惩罚
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

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
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
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
    """方案B优化版v2：几何瞄准与增强优化 Agent
    
    核心策略：
    1. 几何计算：精确计算球到袋口的瞄准点
    2. 物理模拟：验证击球可行性，避免犯规
    3. 智能选择：优先选择成功率高的目标球
    4. 安全击球：严格检查首球犯规和黑8风险
    5. 多重验证：增强首球检测和白球落袋检测
    """
    
    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575  # 标准台球半径（米）
        
        # 优化参数
        self.SIMULATION_COUNT = 30  # 增加模拟次数
        self.SIMULATION_TIMEOUT = 2  # 单次模拟超时
        self.POWER_REDUCTION = 0.9  # 力度衰减系数，降低白球落袋概率
        
        # 噪声参数（与poolenv保持一致）
        self.noise_std = {
            'V0': 0.1,      # 速度标准差
            'phi': 0.1,     # 水平角度标准差（度）
            'theta': 0.1,   # 垂直角度标准差（度）
            'a': 0.003,     # 横向偏移标准差
            'b': 0.003      # 纵向偏移标准差
        }
        # 评估加速：两阶段评估
        # - SCREEN: 用更少的带噪声采样做快速筛选（用于搜索/微调）
        # - FINAL: 对最终候选用全量采样复核，保持最终精度
        self.ROBUSTNESS_SAMPLES_FINAL = 10
        self.ROBUSTNESS_SAMPLES_SCREEN = 4
        self.ROBUST_TOPK = 6  # 仅对 Top-K 候选做全量鲁棒性评估

        # 综合评分权重（保持原先“模拟优先”的倾向）
        self.GEO_WEIGHT = 0.3
        self.SIM_WEIGHT = 0.7

        logger.info("NewAgent (优化版v3：几何瞄准+噪声鲁棒性评估) 已初始化")
    
    def _get_ball_position(self, ball):
        """获取球的2D位置"""
        return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]])
    
    def _get_pocket_positions(self, table):
        """从球桌对象获取真实袋口位置"""
        pockets = []
        for pocket in table.pockets.values():
            pos = pocket.center
            pockets.append(np.array([pos[0], pos[1]]))
        return pockets
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _is_break_state(self, balls, my_targets):
        """判断是否为开球局面（球仍在三角架紧密堆叠）。

        说明：环境不会把 hit_count 传给 agent，因此只能从几何形态推断。
        我们用“所有目标球均未进袋 + 15 颗彩球紧密聚类”的条件来识别开球。
        """
        try:
            object_ids = [str(i) for i in range(1, 16)]
            if 'cue' not in balls:
                return False
            if balls['cue'].state.s == 4:
                return False
            for bid in object_ids:
                if bid not in balls or balls[bid].state.s == 4:
                    return False

            obj_positions = np.array([self._get_ball_position(balls[bid]) for bid in object_ids])
            centroid = obj_positions.mean(axis=0)
            bbox = obj_positions.max(axis=0) - obj_positions.min(axis=0)
            bbox_diag = float(np.linalg.norm(bbox))
            max_r = float(np.max(np.linalg.norm(obj_positions - centroid, axis=1)))

            cue_pos = self._get_ball_position(balls['cue'])
            cue_to_centroid = float(np.linalg.norm(cue_pos - centroid))

            # 阈值取保守：开球时目标球团很紧、与白球距离较远。
            # 若球已散开，bbox/max_r 会明显变大。
            return (bbox_diag < 0.28) and (max_r < 0.18) and (cue_to_centroid > 0.45) and (my_targets != ['8'])
        except Exception:
            return False

    def _break_decision(self, balls, my_targets, table):
        """开球快速策略：少量候选 + 极少评估。

        目标：避免在开球局面跑完整的 Top-K * 鲁棒评估流程。
        关键：必须尽量保证首球合法（首碰必须落在 my_targets 内，且不能先碰 8）。
        """
        cue_pos = self._get_ball_position(balls['cue'])
        object_ids = [str(i) for i in range(1, 16)]
        obj_positions = np.array([self._get_ball_position(balls[bid]) for bid in object_ids])
        rack_centroid = obj_positions.mean(axis=0)

        base_phi = self._calculate_shot_angle(cue_pos, rack_centroid)

        # 在 base_phi 附近搜索一个“首碰合法”的角度（几何预测，代价极低）
        best_phi = None
        best_delta = None
        for delta in [0, -1.5, 1.5, -3, 3, -5, 5, -8, 8, -12, 12, -18, 18, -25, 25]:
            phi = (base_phi + delta) % 360
            first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
            if first_contact is None:
                continue
            if first_contact == '8' and my_targets != ['8']:
                continue
            if first_contact in my_targets:
                best_phi = phi
                best_delta = abs(delta)
                break

        if best_phi is None:
            # 实在找不到就用 base_phi（仍比全量搜索快），后续用轻评估挑较安全力度
            best_phi = base_phi
            best_delta = None

        # 少量力度候选（开球不需要精细调 a/b/theta）
        candidates = []
        for v0 in [5.8, 6.4, 7.0]:
            candidates.append({'V0': v0, 'phi': best_phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0})

        # 轻量评估：只做一次无噪声仿真（开球不需要噪声鲁棒评估）
        best_action = candidates[0]
        best_score = -1e9
        for action in candidates:
            fast_score = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=False)
            if fast_score <= -5000:
                continue

            if fast_score > best_score:
                best_score = fast_score
                best_action = action

        logger.info(
            f"[NewAgent] 开球策略: phi={best_action['phi']:.2f}, V0={best_action['V0']:.2f}" +
            (f", delta={best_delta}" if best_delta is not None else "")
        )
        return best_action
    
    def _calculate_aim_point(self, target_pos, pocket_pos):
        """计算瞄准点（ghost ball position）"""
        direction = np.array(target_pos) - np.array(pocket_pos)
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None
        
        direction = direction / distance
        aim_point = np.array(target_pos) + direction * (2 * self.BALL_RADIUS)
        return aim_point
    
    def _calculate_shot_angle(self, cue_pos, aim_point):
        """计算击球角度（phi）"""
        direction = np.array(aim_point) - np.array(cue_pos)
        angle_rad = np.arctan2(direction[1], direction[0])
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg
    
    def _calculate_shot_power(self, distance, target_to_pocket):
        """根据距离计算击球力度（优化版：降低力度减少白球落袋）"""
        total_dist = distance + target_to_pocket
        
        # 降低力度系数，减少白球落袋风险
        if total_dist < 0.4:
            power = 1.5  # 近距离轻打
        elif total_dist < 0.8:
            power = 2.2  # 中近距离
        elif total_dist < 1.2:
            power = 3.0  # 中距离
        elif total_dist < 1.8:
            power = 4.0  # 中远距离
        else:
            power = 4.8  # 远距离
        
        return power * self.POWER_REDUCTION
    
    def _check_path_clear(self, start_pos, end_pos, balls, exclude_ids):
        """检查路径上是否有障碍球"""
        line_vec = np.array(end_pos) - np.array(start_pos)
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return True
        
        line_vec = line_vec / line_length
        
        for ball_id, ball in balls.items():
            if ball_id in exclude_ids or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - np.array(start_pos)
            projection = np.dot(to_ball, line_vec)
            
            if 0 < projection < line_length:
                closest_point = np.array(start_pos) + line_vec * projection
                distance = np.linalg.norm(ball_pos - closest_point)
                
                if distance < 2.2 * self.BALL_RADIUS:
                    return False
        return True
    
    def _get_first_contact_ball(self, cue_pos, phi, balls):
        """预测白球首先接触的球（精确版）"""
        # 计算击球方向
        phi_rad = np.radians(phi)
        direction = np.array([np.cos(phi_rad), np.sin(phi_rad)])
        
        candidates = []
        
        for ball_id, ball in balls.items():
            if ball_id == 'cue' or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - np.array(cue_pos)
            
            # 投影到击球方向
            proj = np.dot(to_ball, direction)
            if proj <= 0:
                continue  # 球在后方
            
            # 计算垂直距离
            perp_dist = np.abs(np.cross(direction, to_ball))
            
            # 考虑碰撞半径（两球半径之和）
            collision_radius = 2 * self.BALL_RADIUS
            
            if perp_dist < collision_radius:
                # 计算实际碰撞点的距离
                # 使用勾股定理计算碰撞点
                if collision_radius**2 - perp_dist**2 > 0:
                    adjust = np.sqrt(collision_radius**2 - perp_dist**2)
                    collision_dist = proj - adjust
                    if collision_dist > 0:
                        candidates.append((collision_dist, ball_id))
        
        if not candidates:
            return None
        
        # 返回最近的球
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _will_hit_eight_ball(self, cue_pos, phi, balls, target_id):
        """检查是否会在击中目标球之前或之后击中8号球"""
        if '8' not in balls or balls['8'].state.s == 4:
            return False
        
        first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
        
        # 如果首球就是8号球，而目标不是8号球
        if first_contact == '8' and target_id != '8':
            return True
        
        # 如果首球不是目标球，可能会产生连锁碰撞到8号球
        if first_contact is not None and first_contact != target_id:
            # 这种情况下可能会犯规
            return True
        
        return False
    
    def _evaluate_shot_quality(self, cue_pos, target_id, target_pos, pocket_pos, balls, my_targets):
        """评估击球质量（综合评分）- 增强版"""
        score = 100.0
        
        # 计算瞄准点
        aim_point = self._calculate_aim_point(target_pos, pocket_pos)
        if aim_point is None:
            return -1000
        
        # 1. 距离因素
        cue_to_aim = self._calculate_distance(cue_pos, aim_point)
        target_to_pocket = self._calculate_distance(target_pos, pocket_pos)
        total_distance = cue_to_aim + target_to_pocket
        score -= total_distance * 15
        
        # 2. 角度因素（切角难度）
        vec1 = np.array(aim_point) - np.array(cue_pos)
        vec2 = np.array(pocket_pos) - np.array(target_pos)
        
        if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            cos_angle = np.dot(vec1, vec2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            if angle < 20:
                score += 40  # 几乎直球，非常容易
            elif angle < 40:
                score += 20
            elif angle < 60:
                score += 0
            elif angle < 80:
                score -= 20
            else:
                score -= 50  # 大角度很难
        
        # 3. 路径清晰度检查
        # 白球到瞄准点
        if not self._check_path_clear(cue_pos, aim_point, balls, ['cue', target_id]):
            score -= 60
        
        # 目标球到袋口
        if not self._check_path_clear(target_pos, pocket_pos, balls, ['cue', target_id]):
            score -= 40
        
        # 4. 袋口距离奖励
        if target_to_pocket < 0.2:
            score += 50  # 非常接近袋口
        elif target_to_pocket < 0.4:
            score += 30
        elif target_to_pocket < 0.6:
            score += 10
        
        # 5. 首球检查（关键！）
        phi = self._calculate_shot_angle(cue_pos, aim_point)
        first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
        
        # 确定当前应该打什么
        remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        should_hit_eight = len(remaining_own) == 0
        
        if first_contact is not None and first_contact != target_id:
            # 首球不是目标球
            if should_hit_eight:
                # 应该打8号球，但首球不是8号球
                if first_contact != '8':
                    score -= 250  # 严重犯规
            else:
                # 还有目标球，首球不是己方球
                if first_contact not in my_targets or first_contact == '8':
                    score -= 250  # 严重惩罚：首球犯规
        
        # 6. 黑8风险检查
        if target_id != '8' and '8' in [b for b in balls if balls[b].state.s != 4]:
            eight_pos = self._get_ball_position(balls['8'])
            # 检查击球路线是否经过黑8
            if not self._check_path_clear(cue_pos, aim_point, balls, ['cue', target_id, '8']):
                # 路线经过黑8附近
                dist_to_eight = self._calculate_distance(aim_point, eight_pos)
                if dist_to_eight < 3 * self.BALL_RADIUS:
                    score -= 150  # 有撞到黑8的风险
            
            # 额外检查：目标球到袋口的路线是否经过黑8
            if not self._check_path_clear(target_pos, pocket_pos, balls, ['cue', target_id, '8']):
                dist_eight_to_line = self._calculate_distance(eight_pos, pocket_pos)
                if dist_eight_to_line < 4 * self.BALL_RADIUS:
                    score -= 100  # 打进目标球可能连带打进黑8
        
        return score
    
    def _simulate_and_evaluate(self, action, balls, my_targets, table, add_noise=False):
        """模拟击球并详细评估结果（增强版：支持噪声模拟）
        
        参数：
            add_noise: 是否添加噪声模拟真实环境
        """
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            # 如果启用噪声，添加高斯扰动
            if add_noise:
                noisy_action = {
                    'V0': action['V0'] + np.random.normal(0, self.noise_std['V0']),
                    'phi': action['phi'] + np.random.normal(0, self.noise_std['phi']),
                    'theta': action['theta'] + np.random.normal(0, self.noise_std['theta']),
                    'a': action['a'] + np.random.normal(0, self.noise_std['a']),
                    'b': action['b'] + np.random.normal(0, self.noise_std['b'])
                }
                # 限制参数在合理范围内
                noisy_action['V0'] = np.clip(noisy_action['V0'], 0.5, 8.0)
                noisy_action['phi'] = noisy_action['phi'] % 360
                noisy_action['theta'] = np.clip(noisy_action['theta'], 0, 90)
                noisy_action['a'] = np.clip(noisy_action['a'], -0.5, 0.5)
                noisy_action['b'] = np.clip(noisy_action['b'], -0.5, 0.5)
                actual_action = noisy_action
            else:
                actual_action = action
            
            shot.cue.set_state(
                V0=actual_action['V0'],
                phi=actual_action['phi'],
                theta=actual_action['theta'],
                a=actual_action['a'],
                b=actual_action['b']
            )
            
            if not simulate_with_timeout(shot, timeout=self.SIMULATION_TIMEOUT):
                return -200
            
            # 分析结果
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            score = 0
            
            # 确定当前应该打什么球
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            should_hit_eight = len(remaining_own) == 0
            
            # 严重犯规检查 - 白球落袋
            if 'cue' in new_pocketed:
                if '8' in new_pocketed:
                    return -10000  # 白球+黑8同时进袋，一票否决，绝对禁止
                return -1000  # 白球进袋（大幅提高惩罚，活着比进球重要）
            
            # 黑8处理 - 一票否决制
            if '8' in new_pocketed:
                if should_hit_eight:
                    return 250  # 合法打进黑8，获胜！
                else:
                    return -10000  # 误打黑8，一票否决，绝对禁止执行此动作
            
            # 首球犯规检查（最重要的检查）
            first_contact_ball_id = None
            valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
            
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if 'ball' in et.lower() and 'ball' in et.lower():  # ball-ball collision
                    if 'cue' in ids:
                        other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                        if other_ids:
                            first_contact_ball_id = other_ids[0]
                            break
                # 兼容其他事件类型格式
                if ('cushion' not in et) and ('pocket' not in et) and ('spin' not in et) and ('rolling' not in et) and ('sliding' not in et) and ('stationary' not in et):
                    if 'cue' in ids:
                        other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                        if other_ids:
                            first_contact_ball_id = other_ids[0]
                            break
            
            # 首球犯规判定
            if first_contact_ball_id is not None:
                if should_hit_eight:
                    # 只剩黑8，首球必须是黑8
                    if first_contact_ball_id != '8':
                        score -= 180  # 首球犯规（严重）
                else:
                    # 还有目标球，首球必须是己方球（非8号）
                    if first_contact_ball_id not in my_targets:
                        score -= 180  # 首球犯规
                    elif first_contact_ball_id == '8':
                        score -= 180  # 不能先打8号球
            else:
                # 没有碰到任何球
                score -= 120
            
            # 进球得分
            own_pocketed = [bid for bid in new_pocketed if bid in my_targets and bid != '8']
            enemy_pocketed = [bid for bid in new_pocketed if bid not in my_targets and bid not in ['cue', '8']]
            
            score += len(own_pocketed) * 100  # 提高进球奖励
            score -= len(enemy_pocketed) * 40
            
            # 未进球时检查碰库（关键！）
            if len(new_pocketed) == 0:
                cue_hit_cushion = False
                target_hit_cushion = False
                
                for e in shot.events:
                    et = str(e.event_type).lower()
                    ids = list(e.ids) if hasattr(e, 'ids') else []
                    if 'cushion' in et:
                        if 'cue' in ids:
                            cue_hit_cushion = True
                        if first_contact_ball_id is not None and first_contact_ball_id in ids:
                            target_hit_cushion = True
                
                # 未进球且未碰库 = 犯规
                if first_contact_ball_id is not None and not cue_hit_cushion and not target_hit_cushion:
                    score -= 200  # 严厉惩罚未碰库犯规
            
            # 无进球但合法（碰库了）
            if score == 0 and 'cue' not in new_pocketed and '8' not in new_pocketed:
                score = 10
            
            # 袋口斥力场检测：白球停在袋口边缘是极度危险的
            if 'cue' not in new_pocketed and 'cue' in sim_balls:
                cue_final_pos = self._get_ball_position(sim_balls['cue'])
                pockets = self._get_pocket_positions(sim_table)
                
                min_pocket_dist = float('inf')
                for pocket_pos in pockets:
                    dist = self._calculate_distance(cue_final_pos, pocket_pos)
                    if dist < min_pocket_dist:
                        min_pocket_dist = dist
                
                # 袋口危险区域：1.5倍球半径以内
                danger_radius = 1.5 * self.BALL_RADIUS
                if min_pocket_dist < danger_radius:
                    # 距离越近惩罚越重
                    danger_penalty = int(500 * (1 - min_pocket_dist / danger_radius))
                    score -= danger_penalty
                    logger.debug(f"[袋口斥力] 白球距袋口 {min_pocket_dist*1000:.1f}mm，扣除 {danger_penalty} 分")
            
            return score
            
        except Exception as e:
            logger.error(f"[NewAgent] 模拟失败: {e}")
            return -200
    
    def _evaluate_with_robustness(self, action, balls, my_targets, table, samples=None):
        """带噪声的鲁棒性评估（多次采样取平均/最小值）
        
        对同一动作进行多次带噪声模拟，评估其在真实环境中的可靠性。
        使用保守策略：如果有任何一次模拟出现严重犯规，大幅降低评分。
        """
        if samples is None:
            samples = self.ROBUSTNESS_SAMPLES_FINAL

        scores = []
        fatal_errors = []  # 记录致命错误的具体得分

        for i in range(samples):
            score = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=True)
            scores.append(score)
            
            # 统计致命错误（黑8误入、白球落袋）
            if score <= -5000:  # 一票否决级别的错误
                fatal_errors.append((i + 1, score))
                # 早停：一旦出现致命错误，立即拒绝该动作（大幅加速失败候选的筛除）
                logger.warning(
                    f"[鲁棒性检测] 动作在第{i + 1}/{samples}次模拟出现致命错误: {score}，提前拒绝"
                )
                return -10000
        
        # 保守策略：使用平均分和最低分的加权组合
        # 这样既考虑平均表现，又惩罚高风险动作
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        
        # 60%平均分 + 40%最低分，更加保守（提高最低分权重）
        robust_score = avg_score * 0.6 + min_score * 0.4
        
        # 额外检查：如果有多次高风险错误（白球落袋、首球犯规等），大幅降低评分
        high_risk_errors = sum(1 for s in scores if s <= -500)  # 包含白球落袋(-1000)和袋口危险
        high_risk_threshold = max(2, int(math.ceil(samples * 0.2)))
        if high_risk_errors >= high_risk_threshold:
            logger.warning(f"[鲁棒性检测] 动作有{high_risk_errors}/{samples}次高风险错误（≤-500），大幅降低评分")
            robust_score -= 300  # 大幅惩罚
        
        # 中等风险错误检测
        medium_risk_errors = sum(1 for s in scores if -500 < s <= -200)
        medium_risk_threshold = max(4, int(math.ceil(samples * 0.4)))
        if medium_risk_errors >= medium_risk_threshold:
            logger.warning(f"[鲁棒性检测] 动作有{medium_risk_errors}/{samples}次中等错误（-500~-200），降低评分")
            robust_score -= 150
        
        return robust_score
    
    def _find_best_shot(self, balls, my_targets, table):
        """寻找最佳击球方案"""
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return None, -1000, None
        
        cue_pos = self._get_ball_position(cue_ball)
        pockets = self._get_pocket_positions(table)
        
        best_action = None
        best_score = -1000
        best_details = None

        # 第一阶段：对全部候选做“快速单次仿真”筛选（不加噪声），减少昂贵的鲁棒评估次数
        candidates = []
        
        # 确定实际目标球（排除已进袋的）
        active_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            
            for pocket_pos in pockets:
                # 几何评估
                geo_score = self._evaluate_shot_quality(
                    cue_pos, target_id, target_pos, pocket_pos, balls, my_targets
                )
                
                if geo_score > best_score - 30:  # 只考虑较好的方案
                    aim_point = self._calculate_aim_point(target_pos, pocket_pos)
                    if aim_point is None:
                        continue
                    
                    phi = self._calculate_shot_angle(cue_pos, aim_point)
                    cue_to_aim = self._calculate_distance(cue_pos, aim_point)
                    target_to_pocket = self._calculate_distance(target_pos, pocket_pos)
                    V0 = self._calculate_shot_power(cue_to_aim, target_to_pocket)
                    
                    action = {
                        'V0': V0,
                        'phi': phi,
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }

                    # 快速单次仿真（不加噪声）用于筛选
                    sim_score_fast = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=False)
                    if sim_score_fast <= -5000:
                        continue

                    total_fast = geo_score * self.GEO_WEIGHT + sim_score_fast * self.SIM_WEIGHT
                    candidates.append((total_fast, action, target_id, pocket_pos, geo_score, sim_score_fast))

        if not candidates:
            return None, -1000, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[: self.ROBUST_TOPK]

        # 第二阶段：仅对 Top-K 做全量鲁棒评估，保持最终精度
        for _, action, target_id, pocket_pos, geo_score, sim_score_fast in top_candidates:
            sim_score_robust = self._evaluate_with_robustness(
                action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
            )
            total_score = geo_score * self.GEO_WEIGHT + sim_score_robust * self.SIM_WEIGHT

            if total_score > best_score:
                best_score = total_score
                best_action = action
                best_details = {
                    'target_id': target_id,
                    'pocket': pocket_pos,
                    'geo_score': geo_score,
                    'sim_score_fast': sim_score_fast,
                    'sim_score': sim_score_robust,
                }
        
        return best_action, best_score, best_details
    
    def _refine_shot_with_simulation(self, base_action, balls, my_targets, table):
        """使用物理模拟微调击球参数（增强版：更多安全检查）"""
        best_action = base_action.copy()
        # 微调阶段用更少采样做筛选（加速），最后对最优动作做全量采样复核（保精度）
        best_score_screen = self._evaluate_with_robustness(
            base_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_SCREEN
        )
        
        cue_ball = balls.get('cue')
        if cue_ball:
            cue_pos = self._get_ball_position(cue_ball)
        else:
            best_score_final = self._evaluate_with_robustness(
                best_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
            )
            return best_action, best_score_final
        
        for i in range(self.SIMULATION_COUNT):
            # 微调参数（减小搜索范围以提高精度）
            test_action = {
                'V0': base_action['V0'] + np.random.uniform(-0.5, 0.5),
                'phi': base_action['phi'] + np.random.uniform(-4, 4),
                'theta': np.random.uniform(0, 12),
                'a': np.random.uniform(-0.12, 0.12),
                'b': np.random.uniform(-0.12, 0.12)
            }
            
            # 限制力度范围（更保守）
            test_action['V0'] = np.clip(test_action['V0'], 1.2, 5.5)
            test_action['phi'] = test_action['phi'] % 360
            test_action['theta'] = np.clip(test_action['theta'], 0, 15)
            
            # 快速首球检查（在模拟之前）
            first_contact = self._get_first_contact_ball(cue_pos, test_action['phi'], balls)
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            
            # 检查首球是否合法
            if len(remaining_own) > 0:
                # 还有目标球，首球必须是己方球
                if first_contact is not None and first_contact not in my_targets:
                    continue  # 跳过这个参数组合
                if first_contact == '8':
                    continue  # 不能先打8号球
            else:
                # 只剩8号球，首球必须是8号球
                if first_contact is not None and first_contact != '8':
                    continue
            
            # 使用筛选级鲁棒性评估（更少采样）
            score_screen = self._evaluate_with_robustness(
                test_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_SCREEN
            )
            
            if score_screen > best_score_screen:
                best_action = test_action.copy()
                best_score_screen = score_screen
                logger.info(f"[NewAgent] 找到更优方案 (筛选鲁棒得分: {score_screen:.1f})")

        best_score_final = self._evaluate_with_robustness(
            best_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
        )
        logger.info(
            f"[NewAgent] 模拟优化完成 ({self.SIMULATION_COUNT}次搜索), 最终鲁棒得分: {best_score_final:.1f}"
        )
        return best_action, best_score_final
    
    def _try_kick_shot(self, balls, my_targets, table, cue_pos, active_targets):
        """尝试大力解球（Kick Shot）策略
        
        当常规防守无法找到安全方案时，尝试用大力击球依靠复杂碰撞创造机会。
        关键：必须确保首球合法，即使后续轨迹复杂也要遵守规则。
        """
        logger.info("[NewAgent] 尝试大力解球策略（Kick Shot）")
        
        best_kick = None
        best_kick_score = -1000
        
        # 策略1：向最近的合法目标球大力击打，依靠多次碰撞和碰库
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            phi = self._calculate_shot_angle(cue_pos, target_pos)
            
            # 验证首球合法性
            first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
            if first_contact != target_id:
                continue  # 首球不合法，跳过
            
            # 大力击球：V0 = 5.0~6.5，制造复杂局面
            for v0 in [5.0, 5.5, 6.0, 6.5]:
                # 微调角度，避免直接进袋白球
                for delta_phi in [0, -3, 3, -6, 6]:
                    test_phi = (phi + delta_phi) % 360
                    test_first = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    if test_first != target_id:
                        continue  # 首球不合法
                    
                    kick_action = {
                        'V0': v0,
                        'phi': test_phi,
                        'theta': np.random.uniform(0, 8),  # 轻微跳球增加不可预测性
                        'a': 0.0,
                        'b': 0.0
                    }
                    
                    # 使用鲁棒性评估（但降低标准，只要不致命即可）
                    score = self._evaluate_with_robustness(kick_action, balls, my_targets, table)
                    
                    # 大力解球的接受标准：不出现致命错误即可
                    if score > -5000 and score > best_kick_score:
                        best_kick = kick_action
                        best_kick_score = score
                        logger.info(f"[Kick Shot] 找到可行方案: V0={v0:.1f}, phi={test_phi:.1f}, 得分={score:.1f}")
        
        # 策略2：如果直线大力不行，尝试碰库后击打（Bank Shot）
        if best_kick is None or best_kick_score < -200:
            logger.info("[NewAgent] 尝试碰库解球（Bank Shot）")
            # 尝试向球台边缘方向大力击打，利用碰库改变轨迹
            pockets = self._get_pocket_positions(table)
            for pocket_pos in pockets:
                # 计算朝向袋口附近（但不是直接瞄准）的角度
                direction = np.array(pocket_pos) - np.array(cue_pos)
                base_phi = np.degrees(np.arctan2(direction[1], direction[0]))
                if base_phi < 0:
                    base_phi += 360
                
                # 偏移角度确保会碰库
                for offset in [-25, -20, -15, 15, 20, 25]:
                    test_phi = (base_phi + offset) % 360
                    first_contact = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    # 检查首球合法性
                    if first_contact is not None and first_contact not in active_targets:
                        continue
                    
                    bank_action = {
                        'V0': 5.5,  # 中等偏大力度
                        'phi': test_phi,
                        'theta': 5.0,  # 轻微跳球
                        'a': 0.0,
                        'b': -0.1  # 轻微低杆，增加碰库后效果
                    }
                    
                    score = self._evaluate_with_robustness(bank_action, balls, my_targets, table)
                    
                    if score > -5000 and score > best_kick_score:
                        best_kick = bank_action
                        best_kick_score = score
                        logger.info(f"[Bank Shot] 找到可行方案: phi={test_phi:.1f}, 得分={score:.1f}")
                        break  # 找到一个就够了
        
        if best_kick is not None:
            logger.info(f"[NewAgent] 采用大力解球，预期得分: {best_kick_score:.1f}")
            return best_kick
        
        return None
    
    def _get_safe_shot(self, balls, my_targets, table):
        """生成安全的防守击球（增强版v2：设置防守底线，绝境时大力解球）"""
        SAFE_SHOT_THRESHOLD = -50  # 防守底线：低于此分数的动作视为不可接受
        
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return self._random_action()
        
        cue_pos = self._get_ball_position(cue_ball)
        
        # 确定实际目标（排除已进袋的）
        active_targets = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        if len(active_targets) == 0:
            active_targets = ['8'] if balls.get('8') and balls['8'].state.s != 4 else []
        
        if not active_targets:
            return self._random_action()
        
        best_action = None
        best_score = -1000
        
        # 遍历所有目标球，找到能够安全击中的
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            
            # 计算直接指向目标球的角度
            phi = self._calculate_shot_angle(cue_pos, target_pos)
            
            # 预测首球
            first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
            
            # 如果首球就是目标球，这是安全的
            if first_contact == target_id:
                # 验证这个击球 - 增加速度余量确保碰库
                dist_to_target = self._calculate_distance(cue_pos, target_pos)
                # 根据距离计算安全速度，确保有足够动能碰库
                safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)  # 最低2.5，加距离补偿
                safe_v0 = min(safe_v0, 4.0)  # 不要太大
                
                action = {
                    'V0': safe_v0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
                
                # 使用鲁棒性评估
                score = self._evaluate_with_robustness(action, balls, my_targets, table)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            else:
                # 首球不是目标球，尝试找其他角度
                # 尝试绕过障碍球的角度
                for delta_phi in [-5, 5, -10, 10, -15, 15, -20, 20, -30, 30]:
                    test_phi = (phi + delta_phi) % 360
                    test_first = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    if test_first == target_id:
                        # 增加速度余量确保碰库
                        dist_to_target = self._calculate_distance(cue_pos, target_pos)
                        safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)
                        safe_v0 = min(safe_v0, 4.0)
                        
                        action = {
                            'V0': safe_v0,
                            'phi': test_phi,
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0
                        }
                        # 使用鲁棒性评估
                        score = self._evaluate_with_robustness(action, balls, my_targets, table)
                        if score > best_score:
                            best_score = score
                            best_action = action
                            break  # 找到一个合法角度就停止
        
        # 如果还是没找到好的方案，尝试更多角度
        if best_action is None or best_score < -50:
            # 尝试所有可能的角度
            for test_phi in range(0, 360, 15):
                first_contact = self._get_first_contact_ball(cue_pos, test_phi, balls)
                
                if first_contact in active_targets:
                    action = {
                        'V0': 2.8,  # 增加速度确保碰库
                        'phi': test_phi,
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }
                    # 使用鲁棒性评估
                    score = self._evaluate_with_robustness(action, balls, my_targets, table)
                    if score > best_score:
                        best_score = score
                        best_action = action
        
        if best_action is None:
            logger.warning("[NewAgent] 无法找到安全击球，使用最小风险策略")
            # 最后的方案：向最近的己方球轻轻一击
            min_dist = float('inf')
            nearest_target = None
            for tid in active_targets:
                if balls.get(tid) and balls[tid].state.s != 4:
                    dist = self._calculate_distance(cue_pos, self._get_ball_position(balls[tid]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_target = tid
            
            if nearest_target:
                target_pos = self._get_ball_position(balls[nearest_target])
                phi = self._calculate_shot_angle(cue_pos, target_pos)
                dist_to_target = self._calculate_distance(cue_pos, target_pos)
                # 确保足够速度碰库
                safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)
                safe_v0 = min(safe_v0, 4.0)
                
                fallback_action = {
                    'V0': safe_v0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
                # 必须对后备方案进行鲁棒性评估，防止误打黑8
                fallback_score = self._evaluate_with_robustness(fallback_action, balls, my_targets, table)
                
                # 只有在评估通过时才使用这个后备方案
                if fallback_score > best_score:
                    best_action = fallback_action
                    best_score = fallback_score
                
                # 如果后备方案也失败（如误打黑8），返回随机动作
                if best_action is None or best_score <= -5000:
                    logger.warning("[NewAgent] 后备方案也不安全，使用随机动作")
                    return self._random_action()
            else:
                return self._random_action()
        
        # 防守底线检查：如果最佳防守得分太低（大概率犯规），尝试大力解球
        if best_action is not None and best_score < SAFE_SHOT_THRESHOLD:
            logger.warning(f"[NewAgent] 防守得分过低 ({best_score:.1f} < {SAFE_SHOT_THRESHOLD})，尝试大力解球策略")
            kick_action = self._try_kick_shot(balls, my_targets, table, cue_pos, active_targets)
            if kick_action is not None:
                return kick_action
            # 大力解球也失败，返回随机动作
            logger.warning("[NewAgent] 大力解球也未找到合法方案，使用随机动作")
            return self._random_action()
        
        # 最终安全检查：确保返回的动作是经过验证的
        if best_action is None or best_score <= -5000:
            logger.warning(f"[NewAgent] 所有安全策略都失败 (best_score={best_score:.1f})，尝试大力解球")
            kick_action = self._try_kick_shot(balls, my_targets, table, cue_pos, active_targets)
            if kick_action is not None:
                return kick_action
            return self._random_action()
        
        logger.info(f"[NewAgent] 使用安全击球策略，预期得分: {best_score:.1f}")
        return best_action
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数（增强版：更多安全验证）"""
        if balls is None or my_targets is None or table is None:
            logger.warning("[NewAgent] 缺少必要信息，使用随机动作")
            return self._random_action()
        
        try:
            # 开球局面：不值得跑完整在线搜索（会浪费大量时间）
            if self._is_break_state(balls, my_targets):
                logger.info("[NewAgent] 检测到开球局面，采用快速开球策略")
                return self._break_decision(balls, my_targets, table)

            # 检查是否需要打8号球
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                logger.info("[NewAgent] 目标球已清空，切换到8号球")
            
            logger.info(f"[NewAgent] 开始决策，目标球: {my_targets}")
            
            # 第一步：几何计算 + 物理验证找最佳方案
            base_action, base_score, details = self._find_best_shot(balls, my_targets, table)
            
            if details:
                logger.info(f"[NewAgent] 最佳方案: 目标球{details['target_id']}, "
                           f"几何={details['geo_score']:.1f}, 模拟={details['sim_score']:.1f}")
            
            # 如果最佳方案得分太低，使用安全策略
            if base_action is None or base_score < -80:
                logger.info(f"[NewAgent] 未找到好方案 (得分: {base_score:.1f})，使用安全策略")
                return self._get_safe_shot(balls, my_targets, table)
            
            # 第二步：物理模拟微调
            refined_action, final_score = self._refine_shot_with_simulation(
                base_action, balls, my_targets, table
            )
            
            # 如果优化后得分仍然很低，使用安全策略
            if final_score < -80:
                logger.info(f"[NewAgent] 优化后得分仍低 ({final_score:.1f})，改用安全策略")
                return self._get_safe_shot(balls, my_targets, table)
            
            # 最终安全验证
            cue_ball = balls.get('cue')
            if cue_ball:
                cue_pos = self._get_ball_position(cue_ball)
                first_contact = self._get_first_contact_ball(cue_pos, refined_action['phi'], balls)
                
                # 检查首球是否合法
                if len(remaining_own) > 0:
                    # 还有目标球
                    if first_contact is not None and (first_contact not in my_targets or first_contact == '8'):
                        logger.warning(f"[NewAgent] 最终验证发现首球不合法 ({first_contact})，改用安全策略")
                        return self._get_safe_shot(balls, my_targets, table)
                else:
                    # 只打8号球
                    if first_contact is not None and first_contact != '8':
                        logger.warning(f"[NewAgent] 最终验证发现首球不是8号球 ({first_contact})，改用安全策略")
                        return self._get_safe_shot(balls, my_targets, table)
            
            logger.info(f"[NewAgent] 最终决策: V0={refined_action['V0']:.2f}, "
                       f"phi={refined_action['phi']:.2f}, theta={refined_action['theta']:.2f}")
            
            return refined_action
            
        except Exception as e:
            logger.error(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()