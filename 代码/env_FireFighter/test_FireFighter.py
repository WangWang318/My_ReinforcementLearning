from env_FireFighter import EnvFireFighter
import numpy as np
import random

"""
    scenario
    firefighter比着火的住宅数目少一个
    firefighter只能灭左右住宅的火
    firefighter只有左右住宅的partial observation
    
    如何设置reward?
    个体的reward 扑火的house的level降低 则reward++
    左右两侧火均扑灭 退出
    
    如何更新action？
    灭火阶段 动力学方程，向所有agent学？ 向左右两侧的agent学(preferred)
    如果火停了 
    
    
"""

def generate_tgt_list(agt_num):
    tgt_list = np.random.rand(agt_num, 2)
    tgt_list /= np.sum(tgt_list, axis=1, keepdims=True)
    return tgt_list


house_num = 5

env = EnvFireFighter(house_num, my_penalty=-1.0, my_reward=1.0)

max_iter = 10
for i in range(max_iter):
    print("iter= ", i)
    print("actual fire level: ", env.firelevel)
    print("observed fire level: ", env.get_obs())
    tgt_list = generate_tgt_list(house_num - 1)
    print("agent target: ", tgt_list)
    mixed_strategy, reward, firelevel = env.step_1(tgt_list)

    print("all_reward: ", reward)
    print("new strategy: ", mixed_strategy)
    print("fire_level: ", firelevel)
    print(" ")
