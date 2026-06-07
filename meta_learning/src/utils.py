import torch
import numpy as np

def get_obs_bounds(env):
    if hasattr(env.observation_space, 'spaces'):
        return env.observation_space['observation'].low, env.observation_space['observation'].high
    return env.observation_space.low, env.observation_space.high

def get_state(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs['achieved_goal'][:2], obs['observation'][:27]])
    return obs

def get_state_dim(env):
    if hasattr(env.observation_space, 'spaces'):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['achieved_goal'].shape[0]
        return min(27, obs_dim) + min(2, goal_dim)  # add x,y from achieved_goal
    return env.observation_space.shape[0]

def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float().to(device)
    return torch.tensor(np.array(x), dtype=torch.float32).to(device)