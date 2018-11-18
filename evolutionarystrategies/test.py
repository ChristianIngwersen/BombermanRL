from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys
import multiprocessing as mp
import numpy as np
import csv
import torch



if __name__ == '__main__':
	impact = {
	'imp_team': [0.01],
	'imp_enemies': [0.15,0.15,0.15],
	'imp_powerup': [0.15]
	}
	evo_strat = EvolutionaryStrategy(Model, Fitness, impact, populationsize=10, learning_rate = 1)
	for i in range(1):
		manager = mp.Manager()
		output = manager.Queue()
		processes = [mp.Process(target=evo_strat.evolution, args=(x, output)) for x in range(evo_strat.populationsize)]
		for p in processes:
			p.start()
		for p in processes:
			p.join()
		results = [output.get() for p in processes]
		#print(results)
		rewards = [r[0] for r in results]
		epsilons = [r[1] for r in results]
		evo_strat.model.update_params(epsilons, rewards, evo_strat.learning_rate)
		print("Done with iteration {}".format(i))
		if (i)%10==0:
			winrate = evo_strat.play_game()
			print("Average win rate over 5 games {}".format(winrate))
			rewardcsv = open("Rewards.csv", "w")  
			winratecsv = open("Winrate.csv", "w")
			with rewardcsv:
				writer = csv.writer(rewardcsv)
				writer.writerow(rewards)
			with winratecsv:
				writer = csv.writer(winratecsv)
				writer.writerow([winrate])
			rewardcsv.close()
			winratecsv.close()
	torch.save(evo_strat.model.policy.state_dict(),'Model.pt')