import torch
import numpy as np


class RolloutBuffer:
    """
    Rollout buffer class. Used in REINFORCE.

    Parameters
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    device : torch.Device
        Device tensor operations are running on.

    Attributes
    ----------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    states : list
        Stored state vectors.
    actions : list
        Stored action vectors.
    log_probs : list
        Stored log probabilities of actions taken.
    rewards : list
        Stored rewards.
    max_idx : int
        Number of stored steps.
    device : torch.Device
        Device tensor operations are running on.

    Methods
    -------
    store(state, action, log_prob, reward)
        Stores current step information.
    get()
        Returns all stored data as tensors.
    clear()
        Resets buffer.
    """

    def __init__(self, state_dim: int, action_dim: int, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

        self.max_idx = 0

    def store(self,
              state: torch.Tensor,
              action: torch.Tensor,
              log_prob: torch.Tensor,
              reward: float):
        """
        Stores relevant data in respective list.

        Parameters
        ----------
        state : torch.Tensor
            State vector.
        action : torch.Tensor
            Action vector.
        log_prob : torch.Tensor
            Log probability of action under current policy.
        reward : float
            Reward for action.
        """
        self.states.append(state.detach().clone())
        self.actions.append(action.detach().clone())
        self.log_probs.append(log_prob.detach().clone())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))

        self.max_idx += 1

    def get(self):
        """
        Returns all stored data as stacked tensors.

        Returns
        -------
        states : torch.Tensor
            All stored states. Size (T, state_dim)
        actions : torch.Tensor
            All stored actions. Size (T, action_dim)
        log_probs : torch.Tensor
            All stored log probs. Size (T,)
        rewards : torch.Tensor
            All stored rewards. Size (T,)
        """
        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device)

        return states, actions, log_probs, rewards

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.max_idx = 0