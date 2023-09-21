import numpy as np
import random
"""
    需要给每个agent设置一个reward
    
    两种想法
    1. 群体中采取 策略1 策略2 的个体比例各自是多少？
    2. （preferred)每个个体采取 策略1 策略2 的概率是多少？ 可以把策略看作mixed strategy /stochastic policy
    
    利用动力学方程
    1. 每个策略初始化
"""

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
    
    to be continued 如果火停了 


"""

class EnvFireFighter(object):
    def __init__(self, house_num, my_reward=1, my_penalty=-2):
        self.house_num = house_num
        self.fighter_num = self.house_num - 1
        self.firelevel = []
        self.my_reward = my_reward
        self.my_penalty = my_penalty
        # 每个agent的reward
        self.figher_reward = np.zeros((self.fighter_num, 1))
        self.reward = 0

        for i in range(self.house_num):
            self.firelevel.append(3)

    # 将 stochastic_policy转化为deterministic policy
    def step_1(self, mixed_strategy):
        target_list = mixed_strategy.argmax(axis=1)
        # target_list (n, 1)
        print(target_list)
        self.step_2(target_list)
        mixed_strategy = self.update_dynamics(mixed_strategy)

        # return  strategy, global_reward, global_observation
        return mixed_strategy, self.get_all_reward(), self.firelevel

    # 重写
    def step_2(self, target_list):    # 0 left, 1 right
        for i in range(self.house_num):
            # house is on fire
            if self.firelevel[i] > 0:
                # neighbor on fire
                if self.is_neighbour_on_fire(i):
                    num, pos = self.how_many_fighters(i, target_list)
                    if num == 0:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 着火扣分
                            if i == 0:
                                self.figher_reward[i] += self.my_penalty
                            elif i == self.house_num - 1:
                                self.figher_reward[i - 1] += self.my_penalty
                            else:
                                self.figher_reward[i] += self.my_penalty
                                self.figher_reward[i - 1] += self.my_penalty

                    elif num == 1:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 着火扣分 扑火的扣的少，没扑火的扣得多
                            # 两头的消防员不扣分

                            # 中间的消防员
                            if i != 0 and i != self.house_num - 1:
                                self.figher_reward[i + pos] += self.my_penalty * 0.2
                                tmp = 0 if pos == -1 else -1
                                self.figher_reward[i + tmp] += self.my_penalty * 1.0

                        if random.random() < 0.6:
                            self.firelevel[i] = self.firelevel[i] - 1
                            # pos添加reward
                            self.figher_reward[i + pos] += self.my_reward
                    else:
                        self.firelevel[i] = 0
                        # 左右智能体添加reward
                        self.figher_reward[i - 1] += self.my_reward  # 左
                        self.figher_reward[i] += self.my_reward     # 右

                else:
                    num, pos = self.how_many_fighters(i, target_list)
                    if num == 0:
                        if random.random() < 0.4:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 着火扣分
                            if i == 0:
                                self.figher_reward[i] += self.my_penalty
                            elif i == self.house_num - 1:
                                self.figher_reward[i - 1] += self.my_penalty
                            else:
                                self.figher_reward[i] += self.my_penalty
                                self.figher_reward[i - 1] += self.my_penalty

                    elif num == 1:
                        if random.random() < 0.4:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 中间的消防员
                            if i != 0 and i != self.house_num - 1:
                                self.figher_reward[i + pos] += self.my_penalty * 0.2
                                tmp = 0 if pos == -1 else -1
                                self.figher_reward[i + tmp] += self.my_penalty * 1.0

                        self.firelevel[i] = self.firelevel[i] - 1
                        # pos添加reward
                        self.figher_reward[i + pos] += self.my_reward

                    else:
                        self.firelevel[i] = 0
                        # 左右智能体添加reward
                        self.figher_reward[i - 1] += self.my_reward  # 左
                        self.figher_reward[i] += self.my_reward  # 右

            else:   # no fire
                if self.is_neighbour_on_fire(i):
                    num, pos = self.how_many_fighters(i, target_list)
                    if num == 0:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 着火扣分
                            if i == 0:
                                self.figher_reward[i] += self.my_penalty
                            elif i == self.house_num - 1:
                                self.figher_reward[i - 1] += self.my_penalty
                            else:
                                self.figher_reward[i] += self.my_penalty
                                self.figher_reward[i - 1] += self.my_penalty
                    elif num == 1:
                        if random.random() < 0.8:
                            self.firelevel[i] = self.firelevel[i] + 1
                            # 中间的消防员
                            if i != 0 and i != self.house_num - 1:
                                self.figher_reward[i + pos] += self.my_penalty * 0.2
                                tmp = 0 if pos == -1 else -1
                                self.figher_reward[i + tmp] += self.my_penalty * 1.0

                        if random.random() < 0.6:
                            self.firelevel[i] = self.firelevel[i] - 1
                            #
                    else:
                        self.firelevel[i] = 0
                        # 左右智能体添加reward
                        self.figher_reward[i - 1] += self.my_reward  # 左
                        self.figher_reward[i] += self.my_reward  # 右
                else:
                    num, pos = self.how_many_fighters(i, target_list)
                    if num == 0:
                        self.firelevel[i] = 0
                    elif num == 1:
                        self.firelevel[i] = 0
                    else:
                        self.firelevel[i] = 0
        self.regulate_fire()
        for i in range(self.house_num):
            self.reward = self.reward - self.firelevel[i]

    def is_neighbour_on_fire(self, index):
        is_on = False
        if index == 0:
            if self.firelevel[1] > 0:
                is_on = True
        elif index == self.house_num - 1:
            if self.firelevel[index - 1] > 0:
                is_on = True
        else:
            if self.firelevel[index - 1] > 0 or self.firelevel[index + 1] > 0:
                is_on = True
        return is_on

    def reset(self):
        self.firelevel = []
        for i in range(self.house_num):
            self.firelevel.append(3)

    def how_many_fighters(self, index, target_list):
        num = 0
        # 参与救援的fighter的相对坐标
        pos = -1
        if index == 0:
            if target_list[0] == 0:
                num = num + 1
                pos = 0
        elif index == self.house_num - 1:
            if target_list[index - 1] == 1:
                num = num + 1
                pos = -1
        else:
            if target_list[index - 1] == 1:
                num = num + 1
                pos = -1
            if target_list[index] == 0:
                num = num + 1
                pos = 0
        # 当num=1时起作用，pos = -1表示左 pos = 0 表示右
        return num, pos

    def regulate_fire(self):
        for i in range(self.house_num):
            if self.firelevel[i] < 0:
                self.firelevel[i] = 0

    def get_obs(self):
        obs = []
        for i in range(self.fighter_num):
            temp = [0, 0]       # [left, right]
            if random.random() < 1 - np.exp(-self.firelevel[i]):
                temp[0] = 1

            if random.random() < 1 - np.exp(-self.firelevel[i + 1]):
                temp[1] = 1
            obs.append(temp)
        return obs

    def get_all_reward(self):
        return self.figher_reward

    def update_dynamics(self, mixed_strategy):
        dx = np.zeros_like(mixed_strategy)

        # # fighters
        # for i in range(self.house_num - 1):
        #     if i == 0:
        #         dx[i] =
        #     elif i == self.house_num - 2:
        #         dx[i] =
        #     else:
        #         dx[i] =

        return mixed_strategy