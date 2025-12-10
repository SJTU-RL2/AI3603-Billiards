import random
import numpy as np


def set_random_seed(enable=False, seed=42):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        enable (bool): 是否启用固定随机种子
        seed (int): 当 enable 为 True 时使用的随机种子
    """
    if enable:
        random.seed(seed)
        np.random.seed(seed)
        print(f"随机种子已设置为: {seed}")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        # numpy 的随机状态重置
        np.random.seed(None)
        print("随机种子已禁用，使用完全随机模式")
