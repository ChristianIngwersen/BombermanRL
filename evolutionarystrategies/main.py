from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys
import multiprocessing as mp
import numpy as np
import csv
import torch


def generate_epsilon(seed, model):
	torch.manual_seed(seed)
	epsilon = {}
	for key, shape in model.shape().items():
		if model.params[key].type() == "torch.FloatTensor":
			epsilon[key] = torch.randn(shape).float()
		elif model.params[key].type() == "torch.LongTensor":
			epsilon[key] = torch.randn(shape).long()
		else:
			epsilon[key] = torch.randn(shape)
	return epsilon


if __name__ == '__main__':
	impact = {
	'imp_team': [0.01],
	'imp_enemies': [0.15, 0.15, 0.15],
	'imp_powerup': [0.15]
	}
	fitness = Fitness(1)
	fitness.render = True
	model = Model(transfer=True)
	fitness.run_game(model, 0, 1, 0)
