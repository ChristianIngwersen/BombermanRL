### model python file
from policy import Policy
from model_pomm import PommNet
from pommerman_script import featurize, get_unflat_obs_space
import numpy as np
import torch
import pommerman
from gym import spaces

class model():
    # TODO: change model to be a neural network
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
        obs_unflat = get_unflat_obs_space(self.num_channels, 11, self.config['rescale']) # 11 is boardsize and is constant
        min_flat_obs = np.concatenate([obs_unflat.spaces[0].low.flatten(), obs_unflat.spaces[1].low])
        max_flat_obs = np.concatenate([obs_unflat.spaces[0].high.flatten(), obs_unflat.spaces[1].high])
        self.observation_space = spaces.Box(min_flat_obs, max_flat_obs)
        self.masks = torch.zeros(1, 1) # Is true if recurrent == False

        #state_list = torch.load(path) # Needed for loading in simple_ffa_run

        self.policy = Policy(PommNet(obs_shape=self.observation_space.shape,**self.nn_kwargs),action_space=spaces.Discrete(6))
        self.params = self.policy.state_dict()
        self.recurrent_hidden_state = 1

    # TODO: change update params to allow for updateing of neural network.
    def updateparams(self, epsilon, rewards, learningrate):
        best_reward = np.argmax(rewards)
        for key,weights in epsilon[best_reward].items():
            self.params[key] += weights

    def shape(self):
        shape_dict = {}
        for key,weights in self.params.items():
            shape_dict[key] = weights.shape
        return shape_dict

    # TODO: function to mimic agents and to take actions. Look at the code from the guy.
    def act(self, state):
        new_obs = featurize(obs,self.config)
        _, action, _, self.recurrent_hidden_state = self.policy.act(new_obs, self.recurrent_hidden_state, self.masks)
        return action.numpy()