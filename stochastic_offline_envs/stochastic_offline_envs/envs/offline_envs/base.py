from stochastic_offline_envs.samplers.trajectory_sampler import TrajectorySampler
from os import path
import pickle
import os
import json
import numpy as np


class BaseOfflineEnv:

    def __init__(self, p, env_cls, data_policy, horizon, n_interactions, test=False):
        self.env_cls = env_cls
        self.data_policy = data_policy
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.p = p
        if test:
            return

        if self.p is not None and path.exists(self.p):
            print('Dataset file found. Loading existing trajectories.')
            with open(self.p, 'rb') as file:
                self.trajs = pickle.load(file)
        else:
            print('Dataset file not found. Generating trajectories.')
            self.generate_and_save()

    def generate_and_save(self):
        self.trajs = self.collect_trajectories()

        if self.p is not None:
            os.makedirs(path.dirname(self.p), exist_ok=True)
            
            # 保存pickle格式
            with open(self.p, 'wb') as file:
                pickle.dump(self.trajs, file)
                print('Saved trajectories to pickle file.')
            
            # 同时保存JSON格式用于检查
            json_path = self.p.replace('.ds', '.json')
            self.save_as_json(self.trajs, json_path)
            print(f'Saved trajectories to JSON file: {json_path}')
    
    def save_as_json(self, trajs, json_path, max_trajs=100):
        """将轨迹保存为JSON格式，便于检查"""
        
        def convert_to_serializable(obj):
            """将对象转换为JSON可序列化的格式"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        json_data = {
            "metadata": {
                "total_trajectories": len(trajs),
                "saved_trajectories": min(max_trajs, len(trajs)),
                "description": "Toy variant environment trajectories for inspection"
            },
            "trajectories": []
        }
        
        # 只保存前max_trajs个轨迹，避免文件过大
        for i, traj in enumerate(trajs[:max_trajs]):
            traj_data = {
                "trajectory_id": i,
                "length": len(traj.obs),
                "observations": [convert_to_serializable(obs) for obs in traj.obs],
                "actions": convert_to_serializable(traj.actions),
                "rewards": convert_to_serializable(traj.rewards),
                "infos": convert_to_serializable(traj.infos),
                "policy_infos": convert_to_serializable(traj.policy_infos)
            }
            json_data["trajectories"].append(traj_data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def collect_trajectories(self):
        data_policy = self.data_policy()
        sampler = TrajectorySampler(env_cls=self.env_cls,
                                    policy=data_policy,
                                    horizon=self.horizon)
        trajs = sampler.collect_trajectories(self.n_interactions)
        return trajs


def default_path(name, is_data=True):
    # Get the path of the current file
    file_path = path.dirname(path.realpath(__file__))
    # Go up 3 directories
    root_path = path.abspath(path.join(file_path, '..', '..', '..'))
    if is_data:
        # Go to offline data directory
        full_path = path.join(root_path, 'offline_data')
    else:
        full_path = root_path
    # Append the name of the dataset
    return path.join(full_path, name)
