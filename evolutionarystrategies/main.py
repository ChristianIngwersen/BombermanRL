from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys
import multiprocessing as mp

import numpy as np
import csv
import torch

from mpi4py import MPI
comm = MPI.COMM_WORLD   # Defines the default communicator
num_procs = comm.Get_size()  # Stores the number of processes in size.
rank = comm.Get_rank()  # Stores the rank (pid) of the current process
stat = MPI.Status()


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


impact = {
'imp_team': [0.01],
'imp_enemies': [0.1,0.1,0.1],
'imp_powerup': [0.02]
}
fitness = Fitness(1)
model = Model()


if rank == 0:
	# Master work
	print ("Master")
	for i in range(num_procs - 1):
		msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
		print(msg)

else:
	# Worker work

	reward = fitness.run_game(model, 0,1,0)
	#reward = 0
	reward_string = "The reward is {} from {}".format(reward, rank)
	comm.send(reward_string, dest=0)
