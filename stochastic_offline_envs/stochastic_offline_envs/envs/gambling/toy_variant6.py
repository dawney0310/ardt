import gym
import numpy as np
from gym import spaces

class ToyStochasticGame(gym.Env):
    """
    Three-layer stochastic joint-action game:
    - P (agent) has 3 actions: 0,1,2 (1 and 2 treated identically)
    - A (adversary) has 2 actions: 0,1
    - State transitions depend on joint action
    - Rewards given immediately in third layer (s5,s6)
    - Observation has 11 dimensions, last one unused
    - Terminal states explicitly listed
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # agent: 0,1,2
        self.adv_action_space = spaces.Discrete(2)  # adversary: 0,1
        # observation: 11 dimensions (s0-s9 + extra unused)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.uint8)
        self.state = 0
        self.reward_list = [25, 10, -15]  # s7,s8,s9

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.uint8)
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs(), None

    def step(self, action):
        reward = 0
        done = False
        adv_action = np.random.choice(2, 1)[0]





        # Layer 1: s0 -> s1,s2,s3,s4
        if self.state == 0:
            if action == 0:
                if adv_action == 0:
                    self.state = np.random.choice([1,2], p=[0.7,0.3])
                else:
                    self.state = np.random.choice([2,3], p=[0.5,0.5])
            elif action == 1:
                if adv_action == 0:
                    self.state = np.random.choice([1,3], p=[0.6,0.4])
                else:
                    self.state = np.random.choice([2,4], p=[0.3,0.7])

        # Layer 2: s1,s2,s3,s4 -> s5,s6
        elif self.state in [1,2,3,4]:
            if self.state == 1:
                self.state = np.random.choice([5,6], p=[0.6,0.4] if action==0 else [0.4,0.6])
            elif self.state == 2:
                self.state = np.random.choice([5,6], p=[0.7,0.3] if adv_action==0 else [0.3,0.7])
            elif self.state == 3:
                if (action, adv_action) == (0,0):
                    self.state = np.random.choice([5,6], p=[0.6,0.4])
                elif (action, adv_action) == (0,1):
                    self.state = np.random.choice([5,6], p=[0.4,0.6])
                elif (action, adv_action) == (1,0):
                    self.state = np.random.choice([5,6], p=[0.5,0.5])
                elif (action, adv_action) == (1,1):
                    self.state = np.random.choice([5,6], p=[0.3,0.7])
            elif self.state == 4:
                self.state = 6

        # Layer 3: s5,s6 -> terminal reward
        elif self.state in [5,6]:
            if self.state == 5:
                if (action, adv_action) == (0,0):
                    self.state = np.random.choice([7,8,9], p=[0.6,0.3,0.1])

                elif (action, adv_action) == (0,1):
                    self.state = np.random.choice([7,8,9], p=[0.5,0.3,0.2])

                elif (action, adv_action) == (1,0):
                    self.state = np.random.choice([7,8,9], p=[0.3,0.3,0.4])

                elif (action, adv_action) == (1,1):
                    self.state = np.random.choice([7,8,9], p=[0.2,0.3,0.5])

            elif self.state == 6:
                if (action, adv_action) == (0,0):
                    self.state = np.random.choice([7,8,9], p=[0.5,0.2,0.3])

                elif (action, adv_action) == (0,1):
                    self.state = np.random.choice([7,8,9], p=[0.4,0.2,0.4])

                elif (action, adv_action) == (1,0):
                    self.state = np.random.choice([7,8,9], p=[0.25,0.2,0.55])

                elif (action, adv_action) == (1,1):
                    self.state = np.random.choice([7,8,9], p=[0.2,0.2,0.6])


            reward = self.reward_list[self.state - 7]
            done = True


        return self.get_obs(), reward, done, False, {"adv_action": adv_action}
