import numpy as np


# replicator equation
def replicator_dynamics(A, x, alpha=0.01):
    # 平均收益
    u_average = np.matmul(np.matmul(x.T, A), x)
    u = np.matmul(A, x)

    # 计算梯度
    delta = u - u_average
    dx = x * delta
    x = x + dx * alpha

    return x


# Smith动力学
def Smith_dynamics(A, x, alpha=0.01):
    dx = np.zeros_like(x)
    u = np.matmul(A, x)
    for i in range(x.size):
        u_i = u[i]
        sum1 = sum2 = 0
        for j in range(x.size):
            if (j != i):
                u_j = u[j]
                delta_ij = np.maximum(u_i - u_j, 0)
                delta_ji = np.maximum(u_j - u_i, 0)

                sum1 += x[j] * delta_ij
                sum2 += delta_ji
        dx[i] = sum1 - x[i] * sum2
    x = x + dx * alpha
    return x


# BNN动力学
def BNN_dynamics(A, x, alpha=0.01):
    dx = np.zeros_like(x)
    u = np.matmul(A, x)
    u_average = np.matmul(np.matmul(x.T, A), x)

    for i in range(x.size):
        u_i = u[i]
        sum1 = sum2 = 0
        delta_ia = np.maximum(u_i - u_average, 0)
        sum1 += delta_ia
        for j in range(x.size):
            u_j = u[j]
            delta_ja = np.maximum(u_j - u_average, 0)
            sum2 += delta_ja
        dx[i] = sum1 - x[i] * sum2
    x = x + dx * alpha
    return x



def plot_dynamics(dynamics_function, A, x_initial, alpha, state_index, iterations=100):
    # 存储迭代步数和状态变量值的列表
    iteration_list = []
    state_values = []

    # 初始化状态向量
    x = x_initial
    # 迭代并记录数据
    for i in range(iterations + 1):
        iteration_list.append(i)
        state_values.append(x[state_index])
        x = dynamics_function(A, x, alpha)

    # 获取动力学函数的名称
    dynamics_name = dynamics_function.__name__

    # 绘制图像
    plt.plot(iteration_list, state_values)
    plt.xlabel("step")
    plt.ylabel(f"x[{state_index}] ")
    plt.title(f"{dynamics_name}")
    plt.grid(True)
    plt.show()
    return iteration_list, state_values

