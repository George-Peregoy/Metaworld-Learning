import torch
import torch.nn as nn
import os
from src.buffer import RolloutBuffer
from src.network import Network
from src.utils import get_state_dim


class AgentREINFORCE:

    def __init__(self, env):
        self.env = env
        self.state_dim = get_state_dim(env)
        self.action_dim = env.action_space.shape[0]
        self.gamma = 0.99
        self.lr = 1e-5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"MOVING AGENT TO CUDA")

        self.policy = Network(
            layer_sizes=[self.state_dim, 100, 100, self.action_dim],
            lr=self.lr,
            output_activation=nn.Tanh
        )
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.policy.to(self.device)
        self.log_std = self.log_std.to(self.device)

        self.buffer = RolloutBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )

        self.critic = Network(
            layer_sizes=[self.state_dim, 100, 100, 1],
            lr=1e-4,  
        )

        self.critic.to(self.device)

    def choose_action(self, state, training: bool = True):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        mean = self.policy(state)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = torch.clamp(action, -1, 1)
            return action, log_prob
        else:
            action = torch.clamp(mean, -1, 1)
            return action

    def update(self, graph: bool | None = None):

        states, actions, log_probs, rewards = self.buffer.get()

        # compute discounted returns
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = self.critic(states).squeeze()
        advantages = returns - values.detach()

        # recompute log probs for graph tracking
        mean = self.policy(states)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs_fresh = dist.log_prob(actions).sum(dim=-1)

        actor_loss = -(log_probs_fresh * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)

        self.policy.update(loss=actor_loss, graph=graph)
        self.critic.update(loss=critic_loss, graph=graph)

    def save(self, path: str | None = None):
        """
        Saves params to path or returns params as a dict.
        """

        if not path:
            params = {
                'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'log_std': self.log_std.data,
            }
            return params

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'log_std': self.log_std.data,
        }, path)
        print(f"Saved to {path}\n")

    def load(self, path: str | None = None, params: dict | None = None):
        """
        Loads params from a path or given params.
        """

        if params:
            checkpoint = params
        else:
            checkpoint = torch.load(path, map_location=self.device)
            print(f"Loaded from {path}\n")

        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.log_std.data = checkpoint['log_std']