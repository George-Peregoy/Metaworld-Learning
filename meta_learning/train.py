import torch
import torch.nn as nn
import numpy as np
import metaworld
import os
import time
import copy
import random
import json

from torch.func import functional_call
from src.agent import AgentREINFORCE


def compute_reinforce_loss(policy, critic, log_std, states, actions, rewards, gamma, 
                           adapted_policy_params=None, adapted_critic_params=None, 
                           adapted_log_std=None):
    """
    Computes REINFORCE loss with advantage baseline.
    If adapted params are provided, uses those for the forward pass.
    """
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)

    if adapted_policy_params is not None:
        mean = functional_call(policy, adapted_policy_params, states_tensor)
    else:
        mean = policy(states_tensor)

    log_std_use = adapted_log_std if adapted_log_std is not None else log_std
    std = log_std_use.exp()
    dist = torch.distributions.Normal(mean, std)
    log_probs = dist.log_prob(actions_tensor).sum(dim=-1)

    if adapted_critic_params is not None:
        values = functional_call(critic, adapted_critic_params, states_tensor).squeeze()
    else:
        values = critic(states_tensor).squeeze()

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs * advantages).mean()
    critic_loss = nn.functional.mse_loss(values, returns)

    return actor_loss, critic_loss


def inner_loop_adapt(agent, states, actions, rewards, alpha=0.1):
    """
    Performs one manual gradient step on policy and critic.
    Returns adapted parameter dicts.
    """
    actor_loss, critic_loss = compute_reinforce_loss(
        agent.policy, agent.critic, agent.log_std,
        states, actions, rewards, agent.gamma
    )

    # policy gradient step
    policy_params = dict(agent.policy.named_parameters())
    policy_grads = torch.autograd.grad(
        actor_loss, policy_params.values(), create_graph=True, allow_unused=True
    )
    adapted_policy_params = {
        name: p - alpha * (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(policy_params.items(), policy_grads)
    }

    # log_std gradient step
    log_std_grad = torch.autograd.grad(
        actor_loss, agent.log_std, create_graph=True, retain_graph=True
    )[0]
    adapted_log_std = agent.log_std - alpha * log_std_grad

    # critic gradient step
    critic_params = dict(agent.critic.named_parameters())
    critic_grads = torch.autograd.grad(
        critic_loss, critic_params.values(), create_graph=True, allow_unused=True,
        retain_graph=True
    )
    adapted_critic_params = {
        name: p - alpha * (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(critic_params.items(), critic_grads)
    }

    return adapted_policy_params, adapted_critic_params, adapted_log_std


def collect_rollouts(env, agent, n_episodes, adapted_policy_params=None, adapted_critic_params=None, adapted_log_std=None, training=True):
    """
    Collects n_episodes of experience.
    If adapted params provided, uses those for action selection.
    Returns states, actions, rewards lists and mean episode reward.
    """
    all_states = []
    all_actions = []
    all_rewards = []
    episode_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        terminated = truncated = False
        ep_rewards = []

        while not (terminated or truncated):
            if adapted_policy_params is not None:
                mean = functional_call(agent.policy, adapted_policy_params, obs)
                log_std_use = adapted_log_std if adapted_log_std is not None else agent.log_std
                std = log_std_use.exp()
                dist = torch.distributions.Normal(mean, std)
                if training:
                    action = dist.sample()
                else:
                    action = mean
                action = torch.clamp(action, -1, 1)
            else:
                if training:
                    action, _ = agent.choose_action(obs, training=True)
                else:
                    action = agent.choose_action(obs, training=False)

            all_states.append(obs)
            all_actions.append(action.detach())

            obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32)
            ep_rewards.append(reward)
            all_rewards.append(reward)

        episode_rewards.append(sum(ep_rewards))

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    return all_states, all_actions, all_rewards, mean_reward


def ml1(env_name):

    seed = 42
    ml1_benchmark = metaworld.ML1('reach-v3', seed=seed)

    train_envs = ml1_benchmark.train_classes['reach-v3']()
    agent = AgentREINFORCE(env=train_envs)

    meta_iterations = 500
    adapt_episodes = 10
    meta_batch_size = 20
    alpha = 0.01

    root_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(root_dir, f"checkpoints/ML1/ml1.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metric_path = os.path.join(root_dir, f"metrics/ML1/ml1.json")
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    videos_path = os.path.join(root_dir, f"videos/ML1/ml1.mp4")
    os.makedirs(os.path.dirname(videos_path), exist_ok=True)

    start = 0
    if os.path.exists(metric_path):
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        start = metrics["iterations"][-1] + 1
        print(f"Resumng data collection from iter {start}")
        best_score = max(
            p + max(g, 0) for p, g in zip(metrics["post_mean"], metrics["gap"])
        )
        print(f"Best score is currently {best_score:.3f}")
    else:
        metrics = {
            "iterations": [],
            "pre_mean": [],
            "pre_std": [],
            "post_mean": [],
            "post_std": [],
            "gap": []
        }
        best_score = -float('inf')

    if os.path.exists(save_path):
        print(f"Resuming from: {save_path}")
        agent.load(path=save_path)

    meta_optim = torch.optim.Adam([
        *agent.policy.parameters(),
        *agent.critic.parameters(),
        agent.log_std,
    ], lr=1e-3)

    for meta_iter in range(start, meta_iterations):
        tasks = random.sample(ml1_benchmark.train_tasks, meta_batch_size)
        pre_reward_list = []
        post_reward_list = []
        meta_loss_total = torch.tensor(0.0)

        for task in tasks:
            train_envs.set_task(task)

            # pre eval 
            _, _, _, pre_mean = collect_rollouts(
                train_envs, agent, adapt_episodes,
                training=False
            )
            pre_reward_list.append(pre_mean)

            # inner loop rollouts under current meta params
            states, actions, rewards, _ = collect_rollouts(
                train_envs, agent, adapt_episodes,
                training=True
            )

            # manual gradient step
            adapted_policy_params, adapted_critic_params, adapted_log_std = inner_loop_adapt(
                agent, states, actions, rewards, alpha=alpha
            )

            # post rollouts using adapted params
            post_states, post_actions, post_rewards, post_mean = collect_rollouts(
                train_envs, agent, adapt_episodes,
                adapted_policy_params=adapted_policy_params,
                adapted_critic_params=adapted_critic_params,
                adapted_log_std=adapted_log_std,
                training=True
            )
            post_reward_list.append(post_mean)

            # meta loss on post rollouts using adapted params
            task_actor_loss, _ = compute_reinforce_loss(
                agent.policy, agent.critic, agent.log_std,
                post_states, post_actions, post_rewards, agent.gamma,
                adapted_policy_params=adapted_policy_params,
                adapted_critic_params=adapted_critic_params,
                adapted_log_std=adapted_log_std
            )
            meta_loss_total = meta_loss_total + task_actor_loss

        # outer loop update
        meta_loss_total = meta_loss_total / meta_batch_size
        meta_optim.zero_grad()
        meta_loss_total.backward()
        meta_optim.step()

        # metric tracking
        pre_tensor = torch.tensor(pre_reward_list)
        post_tensor = torch.tensor(post_reward_list)

        pre_reward = pre_tensor.mean().item()
        post_reward = post_tensor.mean().item()
        pre_std = pre_tensor.std().item()
        post_std = post_tensor.std().item()
        gap = post_reward - pre_reward

        metrics["iterations"].append(meta_iter)
        metrics["pre_mean"].append(pre_reward)
        metrics["pre_std"].append(pre_std)
        metrics["post_mean"].append(post_reward)
        metrics["post_std"].append(post_std)
        metrics["gap"].append(gap)

        with open(metric_path, 'w') as f:
            json.dump(metrics, f)

        score = post_reward + max(gap, 0)
        print(f"Iter: {meta_iter} | pre: {pre_reward:.3f} | post: {post_reward:.3f} | gap: {gap:.4f} | score: {score:.3f}")
        if score > best_score:
            best_score = score
            agent.save(path=save_path)

def ml10():

    seed = 42
    ml10_benchmark = metaworld.ML10(seed=seed)
    train_classes = ml10_benchmark.train_classes
    train_tasks = ml10_benchmark.train_tasks

    train_envs = {
        name: cls() for name, cls in train_classes.items()
    }

    agent = AgentREINFORCE(env=list(train_envs.values())[0]) # space is equal accross envs

    meta_iterations = 500
    adapt_episodes = 10
    meta_batch_size = 20
    alpha = 0.01

    root_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(root_dir, f"checkpoints/ML10/ml10.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metric_path = os.path.join(root_dir, f"metrics/ML10/ml10.json")
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    videos_path = os.path.join(root_dir, f"videos/ML10/ml10.mp4") # unused for now
    os.makedirs(os.path.dirname(videos_path), exist_ok=True)

    start = 0
    if os.path.exists(metric_path):
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        start = metrics["iterations"][-1] + 1
        print(f"Resumng data collection from iter {start}")
        best_score = max(
            p + max(g, 0) for p, g in zip(metrics["post_mean"], metrics["gap"])
        )
        print(f"Best score is currently {best_score:.3f}")
    else:
        metrics = {
            "iterations": [],
            "pre_mean": [],
            "pre_std": [],
            "post_mean": [],
            "post_std": [],
            "gap": []
        }
        best_score = -float('inf')

    if os.path.exists(save_path):
        print(f"Resuming from: {save_path}")
        agent.load(path=save_path)

    meta_optim = torch.optim.Adam([
        *agent.policy.parameters(),
        *agent.critic.parameters(),
        agent.log_std,
    ], lr=1e-3)

    for meta_iter in range(start, meta_iterations):
        tasks = random.sample(train_tasks, meta_batch_size)
        pre_reward_list = []
        post_reward_list = []
        meta_loss_total = torch.tensor(0.0)

        for task in tasks:
            env = train_envs[task.env_name]
            env.set_task(task)

            # pre eval 
            _, _, _, pre_mean = collect_rollouts(
                env, agent, adapt_episodes,
                training=False
            )
            pre_reward_list.append(pre_mean)

            # inner loop rollouts under current meta params
            states, actions, rewards, _ = collect_rollouts(
                env, agent, adapt_episodes,
                training=True
            )

            # manual gradient step
            adapted_policy_params, adapted_critic_params, adapted_log_std = inner_loop_adapt(
                agent, states, actions, rewards, alpha=alpha
            )

            # post rollouts using adapted params
            post_states, post_actions, post_rewards, post_mean = collect_rollouts(
                env, agent, adapt_episodes,
                adapted_policy_params=adapted_policy_params,
                adapted_critic_params=adapted_critic_params,
                adapted_log_std=adapted_log_std,
                training=True
            )
            post_reward_list.append(post_mean)

            # meta loss on post rollouts using adapted params
            task_actor_loss, _ = compute_reinforce_loss(
                agent.policy, agent.critic, agent.log_std,
                post_states, post_actions, post_rewards, agent.gamma,
                adapted_policy_params=adapted_policy_params,
                adapted_critic_params=adapted_critic_params,
                adapted_log_std=adapted_log_std
            )
            meta_loss_total = meta_loss_total + task_actor_loss

        # outer loop update
        meta_loss_total = meta_loss_total / meta_batch_size
        meta_optim.zero_grad()
        meta_loss_total.backward()
        meta_optim.step()

        # metric tracking
        pre_tensor = torch.tensor(pre_reward_list)
        post_tensor = torch.tensor(post_reward_list)

        pre_reward = pre_tensor.mean().item()
        post_reward = post_tensor.mean().item()
        pre_std = pre_tensor.std().item()
        post_std = post_tensor.std().item()
        gap = post_reward - pre_reward

        metrics["iterations"].append(meta_iter)
        metrics["pre_mean"].append(pre_reward)
        metrics["pre_std"].append(pre_std)
        metrics["post_mean"].append(post_reward)
        metrics["post_std"].append(post_std)
        metrics["gap"].append(gap)

        with open(metric_path, 'w') as f:
            json.dump(metrics, f)

        score = post_reward + max(gap, 0)
        print(f"Iter: {meta_iter} | pre: {pre_reward:.3f} | post: {post_reward:.3f} | gap: {gap:.4f} | score: {score:.3f}")
        if score > best_score:
            best_score = score
            agent.save(path=save_path)

if __name__ == "__main__":

    """env_name = 'reach-v3'
    start = time.perf_counter()
    ml1(env_name)
    end = time.perf_counter()
    print(f"Time to complete ML1 training: {end-start:.2f}")"""

    start = time.perf_counter()
    ml10()
    end = time.perf_counter()
    print(f"Time to complete ML10 training: {end-start:.2f}")