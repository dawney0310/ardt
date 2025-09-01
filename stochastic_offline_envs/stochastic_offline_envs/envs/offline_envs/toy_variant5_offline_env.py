from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.gambling.toy_variant5 import ToyStochasticGame
from stochastic_offline_envs.policies.random import RandomPolicy

class ToyVariant5OfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('toy_variant5.ds'), horizon=5, n_interactions=int(1e5)):
        self.env_cls = lambda: ToyStochasticGame()
        self.test_env_cls = lambda: ToyStochasticGame()

        def data_policy_fn():
            test_env = self.env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, self.env_cls, data_policy_fn, horizon, n_interactions)
