from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys
import multiprocessing as mp



if __name__ == '__main__':
	impact = {
	'imp_team': [0.2],
	'imp_enemies': [0.2,0.2,0.2],
	'imp_powerup': [0.2]
	}
	evo_strat = EvolutionaryStrategy(Model, Fitness, impact, populationsize=15, learning_rate = 5e-4)
	for i in range(5):
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
    #evo_strat.parallel_evolution()
    #if (i-1)%100==0:
    	#print("Average win rate over 5 games {}".format(evo_strat.play_game()))
