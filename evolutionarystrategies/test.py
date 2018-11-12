from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys

evo_strat = EvolutionaryStrategy(Model, Fitness, processes=1, learning_rate = 5e-4)


for i in range(5):
    evo_strat.parallel_evolution()
    if (i-1)%100==0:
    	print("Average win rate over 5 games {}".format(evo_strat.play_game()))

