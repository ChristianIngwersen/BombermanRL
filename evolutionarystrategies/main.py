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
	'imp_enemies': [0.1,0.1,0.1],
	'imp_powerup': [0.02]
	}
	evo_strat = EvolutionaryStrategy(Model, Fitness, impact, populationsize=30, learning_rate = 0.01)
	rewardcsv = open("Rewards.csv", "w")  
	winratecsv = open("Winrate.csv", "w")
	rewardcsv.close()
	winratecsv.close()
	for i in range(100):
		manager = mp.Manager()
		output = manager.Queue()
		processes = [mp.Process(target=evo_strat.evolution, args=(x, output)) for x in range(evo_strat.populationsize)]
		for p in processes:
			p.start()
		for p in processes:
			p.join()
		results = [output.get() for p in processes]
		rewards = [r[0] for r in results]
		epsilons = []
		seed = [epsilons.append(generate_epsilon(r[1], evo_strat.model)) for r in results]
		evo_strat.model.update_params(epsilons, rewards, evo_strat.learning_rate)
		print("Done with iteration {}".format(i))
		if (i)%10==0:
			winrate = evo_strat.play_game(10)
			print("Average win rate over 10 games {}".format(winrate))
			rewardcsv = open("Rewards.csv", "a")  
			winratecsv = open("Winrate.csv", "a")
			with rewardcsv:
				writer = csv.writer(rewardcsv)
				writer.writerow(rewards)
			with winratecsv:
				writer = csv.writer(winratecsv)
				writer.writerow([winrate])
			rewardcsv.close()
			winratecsv.close()
	torch.save(evo_strat.model.policy.state_dict(),'Model.pt')