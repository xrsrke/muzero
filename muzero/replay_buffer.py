# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/08_replay_buffer.ipynb.

# %% auto 0
__all__ = ['MuzeroConfig', 'ReplayBuffer']

# %% ../nbs/08_replay_buffer.ipynb 4
from dataclasses import dataclass

from torch import nn

# %% ../nbs/08_replay_buffer.ipynb 5
@dataclass
class MuzeroConfig:
    # environment
    action_space: int = 1
    observation_space: int = 1
    
    # for replay buffer
    window_size: int = 10
    batch_size: int = 1000

# %% ../nbs/08_replay_buffer.ipynb 6
class ReplayBuffer:
    def __init__(self, config = MuzeroConfig()):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
    
    def save_game(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def sample_batch(self):
        pass
    
    def sample_game(self):
        pass
