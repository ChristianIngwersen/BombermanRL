from fitness import Fitness
from evolutionarystrategy import EvolutionaryStrategy
from model import Model
from mpi4py import MPI
import sys
import torch

comm = MPI.COMM_WORLD   # Defines the default communicator
num_procs = comm.Get_size()  # Stores the number of processes in size.
rank = comm.Get_rank()  # Stores the rank (pid) of the current process
stat = MPI.Status()

impact = {
	'imp_team': [0.01],
	'imp_enemies': [0.1, 0.1, 0.1],
	'imp_powerup': [0.02]
}

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

def slave():
	# Worker work
	evo_strat = EvolutionaryStrategy(Model, Fitness, impact, learning_rate=0.01, transfer = True)

	for i in range(0,100):
		# Play a game and return the reward and seed.
		reward, seed = evo_strat.evolution()
		reward_string = "The reward is {} and seed is {} from worker {}".format(reward, seed, rank)
		comm.send((reward, seed, reward_string), dest=0)

		dict = comm.recv(source = 0)
		evo_strat.model.policy.load_state_dict(dict)


def master():
	import csv
	print ("Master")
	rewardcsv = open("data/Rewards.csv", "w")
	winratecsv = open("data/Winrate.csv", "w")
	rewardcsv.close()
	winratecsv.close()

	evo_strat = EvolutionaryStrategy(Model, Fitness, impact, learning_rate=0.01, transfer= True)

	for i in range(0,100):
		seeds = []
		rewards = []
		for m in range(num_procs - 1):
			msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
			seeds.append(msg[1])
			rewards.append(msg[0])

		epsilons = []
		seed = [epsilons.append(generate_epsilon(s, evo_strat.model)) for s in seeds]
		evo_strat.model.update_params(epsilons, rewards, evo_strat.learning_rate)
		sys.stdout.flush()
		for k in range(num_procs - 1):
			comm.send(evo_strat.model.params, dest = k+1)

		print("Done with iteration {}".format(i))
		if (i) % 10 == 0:
			winrate = evo_strat.play_game(10)
			print("Average win rate over 10 games {}".format(winrate))
			rewardcsv = open("data/Rewards.sv", "a")
			winratecsv = open("data/Winrate.csv", "a")
			with rewardcsv:
				writer = csv.writer(rewardcsv)
				writer.writerow(rewards)
			with winratecsv:
				writer = csv.writer(winratecsv)
				writer.writerow([winrate])
			rewardcsv.close()
			winratecsv.close()


if __name__ == '__main__':
	if rank == 0:
		master()
	else:
		slave()

