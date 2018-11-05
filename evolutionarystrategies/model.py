### model python file
from policy import Policy
from model_pomm import PommNet
from pommerman_script import get_unflat_obs_space
import numpy as np
import torch
from gym import spaces


class Model:

    def __init__(self):
        self.nn_kwargs={
            'batch_norm':  True,
            'recurrent': False,
            'hidden_size': 512,
        } # Found in main.py
        self.config = {
                        'recode_agents': True,
                        'compact_powerups': True,
                        'compact_structure': True,
                        'rescale': True,
                        } # Found in pommerman.py
        self.num_channels = 15 # Found in pommerman.py
        if self.config['recode_agents']:
            self.num_channels -= 2
        if self.config['compact_powerups']:
            self.num_channels -= 2
        if self.config['compact_structure']:
            self.num_channels -= 2
        obs_unflat = get_unflat_obs_space(self.num_channels, 11, self.config['rescale'])  # 11 is boardsize and is constant
        min_flat_obs = np.concatenate([obs_unflat.spaces[0].low.flatten(), obs_unflat.spaces[1].low])
        max_flat_obs = np.concatenate([obs_unflat.spaces[0].high.flatten(), obs_unflat.spaces[1].high])
        self.observation_space = spaces.Box(min_flat_obs, max_flat_obs)
        self.masks = torch.zeros(1, 1)  # Is true if recurrent == False
        self.policy = Policy(PommNet(obs_shape=self.observation_space.shape,**self.nn_kwargs),action_space=spaces.Discrete(6))
        self.params = self.policy.state_dict()
        self.recurrent_hidden_state = 1

    def copy(self):
        copy = Model()
        copy.params = self.params
        copy.policy.load_state_dict(copy.params)
        return copy

    def update_params(self, epsilon, rewards, learning_rate):
        for idx, reward in enumerate(rewards):
            for key,weights in epsilon[idx].items():
                self.params[key] += learning_rate*1/len(rewards)*reward*weights
        self.policy.load_state_dict(self.params)

    def shape(self):
        shape_dict = {}
        for key, weights in self.params.items():
            shape_dict[key] = weights.shape
        return shape_dict

    def act(self, state):
        new_obs = state
        _, action, _, self.recurrent_hidden_state = self.policy.act(new_obs, self.recurrent_hidden_state, self.masks)
        return action.numpy()


