import numpy as np

# a = np.array([1,2,3]).reshape(-1,1)
# print("a:", a)
# b = a ** 2
# print("b:", b)
# c = np.sum(a ** 2, 1)
# print("c:", c)
# print("shape of c:", c.shape)
# d = np.sum(a ** 2, 1).reshape(-1, 1)
# print("d:", d)
# print("shape of d:", d.shape)
# e = c + d
# print("e:", e)
# f = a.T
# print("f:", f)
# g = np.dot(a, f)
# print("g:", g)
# h = np.dot(f, a)
# print("h:", h)
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义SE核函数
# def se_kernel(x1, x2, l=1.0):
#     """Squared Exponential核函数"""
#     dist_sq = np.sum((x1[:, None] - x2[None, :])**2, axis=-1)
#     return np.exp(-dist_sq / (2 * l**2))

# # 1. 选择点集（我们无法处理无限点，所以离散化）
# X = np.linspace(-5, 5, 100).reshape(-1, 1)

# # 2. 计算均值向量（零均值）
# mu = np.zeros(len(X))  # 因为 m(x)=0

# # 3. 计算协方差矩阵
# K = se_kernel(X, X, l=1.0)

# # 4. 从多元正态分布中抽取样本
# np.random.seed(42)
# num_samples = 5
# samples = np.random.multivariate_normal(mu, K, num_samples)

# #5. 这些样本就是不同的 f(x) 函数！
# plt.figure(figsize=(10, 6))
# for i in range(num_samples):
#     plt.plot(X, samples[i], label=f'样本函数 f_{i+1}(x)')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('从 GP(0, SE) 中抽取的样本函数')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# # 比较不同均值函数对样本函数的影响
# plt.figure(figsize=(15, 5))

# # 三个不同的均值函数
# mean_funcs = [
#     (lambda x: 0*x, "零均值: m(x)=0"),
#     (lambda x: 0.5*x, "线性均值: m(x)=0.5x"),
#     (lambda x: 2*np.sin(0.5*x), "正弦均值: m(x)=2sin(0.5x)")
# ]

# for idx, (m_func, title) in enumerate(mean_funcs, 1):
#     plt.subplot(1, 3, idx)
    
#     # 计算均值向量
#     mu = m_func(X).flatten()
    
#     # 从 GP(m(x), SE) 中抽取样本
#     samples = np.random.multivariate_normal(mu, K, 3)
    
#     # 绘制样本函数
#     for i in range(3):
#         plt.plot(X, samples[i], alpha=0.7, label=f'f_{i+1}(x)')
    
#     # 绘制均值函数
#     plt.plot(X, mu, 'k--', linewidth=2, label='均值函数 m(x)')
    
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def generate_gp_data(true_func, X, sigma_n=0.1):
#     """
#     生成高斯过程回归的模拟数据
    
#     参数:
#     - true_func: 真实函数 f(x)
#     - X: 输入点
#     - sigma_n: 噪声标准差
    
#     返回:
#     - y: 带噪声的观测值
#     - f_true: 真实函数值（无噪声）
#     """
#     # 1. 计算真实函数值
#     f_true = true_func(X)
#     print("Shape of f_true:", f_true.shape)
    
#     # 2. 添加高斯噪声
#     np.random.seed(42)
#     noise = np.random.normal(0, sigma_n, size=len(X)).reshape(-1,1)
#     print("Shape of noise:", noise.shape)
#     y = f_true + noise
    
#     return y, f_true

# # 定义真实函数（我们假设它来自某个高斯过程先验）
# def true_function(x):
#     """一个平滑的函数"""
#     return np.sin(0.5*x) + 0.2*x

# # 生成数据
# np.random.seed(42)
# n_train = 20
# n_test = 100

# # 训练点（观测数据）
# X_train = np.random.uniform(-5, 5, n_train).reshape(-1, 1)
# y_train, f_train_true = generate_gp_data(true_function, X_train, sigma_n=0.2)

# # 测试点（用于预测）
# X_test = np.linspace(-6, 6, n_test).reshape(-1, 1)
# f_test_true = true_function(X_test)

# # 可视化
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# print("Shape of X_train:", X_train.shape)
# print("Shape of y_train:", y_train.shape)
# plt.scatter(X_train, y_train, c='red', s=50, zorder=10, label='Observation Data (with noise)')
# plt.plot(X_test, f_test_true, 'k--', linewidth=2, label='True Function f(x)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data Generation Process')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# # 显示噪声分布
# x_example = np.array([0.0])
# y_example, f_example = generate_gp_data(true_function, x_example, sigma_n=0.2)

# # 生成多个噪声样本
# n_samples = 1000
# noise_samples = np.random.normal(0, 0.2, n_samples)
# f_val = true_function(x_example)
# y_samples = f_val + noise_samples

# plt.hist(y_samples, bins=30, density=True, alpha=0.7, label='Distribution of Observed y')
# plt.axvline(x=f_val, color='red', linestyle='--', linewidth=2, label=f'True Value f({x_example[0]})')

# # 绘制高斯分布曲线
# from scipy.stats import norm
# x_plot = np.linspace(f_val-1, f_val+1, 100)
# plt.plot(x_plot, norm.pdf(x_plot, f_val, 0.2), 'k-', linewidth=2, label=f'N(f, σ_n²)')

# plt.xlabel('y')
# plt.ylabel('Probability Density')
# plt.title(f'Observation Distribution at x={x_example[0]}')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


def predict_gp(X_obs, y_obs, X_star, kernel):
    """高斯过程回归预测"""
    
    # 计算核矩阵
    K_obs = kernel(X_obs, X_obs)
    K_star = kernel(X_star, X_obs)
    K_star_star = kernel(X_star, X_star)
    
    # 添加噪声项（正则化）
    K_obs_reg = K_obs + 1e-6 * np.eye(len(X_obs))
    
    # 计算后验分布参数
    L = np.linalg.cholesky(K_obs_reg)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    
    # 预测均值
    mu_star = K_star @ alpha
    
    # 预测方差
    v = np.linalg.solve(L, K_star.T)
    cov_star = K_star_star - v.T @ v
    
    return mu_star, cov_star

# 定义SE核函数
def se_kernel(x1, x2, l=1.0):
    """Squared Exponential核函数"""
    # 确保形状正确
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    
    # 计算平方距离
    x1_sq = np.sum(x1**2, axis=1).reshape(-1, 1)
    x2_sq = np.sum(x2**2, axis=1).reshape(1, -1)
    dist_sq = x1_sq + x2_sq - 2 * x1 @ x2.T
    
    return np.exp(-dist_sq / (2 * l**2))

# 创建一些观测数据
np.random.seed(42)
n_obs = 5
X_obs = np.random.uniform(-3, 3, n_obs).reshape(-1, 1)
y_obs = np.sin(X_obs).flatten() + np.random.normal(0, 0.1, n_obs)  # 带噪声的正弦波

# 预测点
X_star = np.linspace(-5, 5, 100).reshape(-1, 1)

# 使用我们的函数进行预测
mu_star, cov_star = predict_gp(X_obs, y_obs, X_star, se_kernel)

# 计算标准差（方差的对角线）
std_star = np.sqrt(np.diag(cov_star))

# 可视化
plt.figure(figsize=(12, 6))

# 绘制观测点
plt.scatter(X_obs, y_obs, c='red', s=100, zorder=10, label='Observations')

# 绘制预测均值
plt.plot(X_star, mu_star, 'b-', linewidth=2, label='Predicted Mean')

# 绘制置信区间（95%置信区间）
plt.fill_between(X_star.flatten(), 
                 mu_star - 1.96*std_star, 
                 mu_star + 1.96*std_star, 
                 alpha=0.3, color='blue', label='95% Confidence Interval')

# 绘制真实函数（用于对比）
X_true = np.linspace(-5, 5, 200)
y_true = np.sin(X_true)
plt.plot(X_true, y_true, 'k--', alpha=0.5, label='True Function (sin(x))')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gaussian Process Regression Prediction Example')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Number of Observations:", n_obs)
print("Number of Prediction Points:", len(X_star))
print(f"Shape of Predicted Mean: {mu_star.shape}")
print(f"Shape of Predicted Covariance: {cov_star.shape}")