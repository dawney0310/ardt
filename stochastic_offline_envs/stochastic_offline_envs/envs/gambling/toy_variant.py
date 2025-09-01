import gym
import numpy as np
from gym import spaces


class ToyStochasticGame(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.adv_action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.uint8  # 多一个特殊终点
        )
        self.state = 0
        # 原始奖励 + 新的 jackpot 状态 (index 11)
        self.reward_list = [5, 5, 0, 6, 1, 2, 7, 4, 100]  

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs(), None

    def step(self, action):
        reward = 0
        done = False
        adv_action = np.random.choice(2, 1)[0]

        if self.state == 0:
            if adv_action == 0:
                self.state = np.random.choice([1, 2], p=[0.6, 0.4]) 
            elif adv_action == 1:
                self.state = np.random.choice([1, 2], p=[0.2, 0.8]) 

            self.state += np.clip(action, 0, 1) * 2

            if self.state in [3, 4]:
                reward = self.reward_list[-3 + adv_action]  # 对应倒数第3/倒数第2
                done = True
        
        elif self.state in [1, 2]:
            # 以极小概率直接跳 jackpot
            if np.random.rand() < 0.02:   # 2% 概率 jackpot
                self.state = 11
                reward = self.reward_list[-1]  # jackpot reward
                done = True
            else:
                # 正常逻辑
                self.state = (3 * self.state + 2 + action)
                reward = self.reward_list[self.state - 5]
                done = True

        return self.get_obs(), reward, done, False, {"adv": adv_action}
