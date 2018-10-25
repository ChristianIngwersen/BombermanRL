import importlib
from policy import Policy
from model_pomm import PommNet
from rollout_storage import RolloutStorage


import torch
import pommerman


class PytorchAgent(pommerman.agents.BaseAgent):

    def __init__(self, character=pommerman.characters.Bomber):
        super(PytorchAgent, self).__init__(character)
        self.policy = None
        self.nn_kwargs={
            'batch_norm':  True,
            'recurrent': False,
            'hidden_size': 512,
        }
        self.recurrent_hidden_state = None
        self.masks = torch.ones(1, 1)

    def act(self, obs, action_space):
    	if self.policy == None:
    		nn = PommNet(obs_shape=(1092,),**self.nn_kwargs)
    		state_list = torch.load('/home/jakob/02456/BombermanRL/PommeFFACompetitionFast-v0.pt')
    		self.policy = Policy(nn,action_space=action_space)
    		self.policy.load_state_dict(state_list[0])
    	rollouts = RolloutStorage(1, 1,(1092,),action_space,1)
    	value, action, _, _ = self.policy.act(rollouts.obs[0], rollouts.recurrent_hidden_states[0], rollouts.masks[0])
    	print(action)
    	return action.data.numpy()