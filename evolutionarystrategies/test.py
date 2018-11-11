from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model


model = Model()
fitness = Fitness()
evo_strat = EvolutionaryStrategy(model, fitness, learning_rate = 5e-4)


for i in range(5):
    evo_strat.parallel_evolution()
    if (i-1)%100==0:
    	print("Average win rate over 5 games {}".format(fitness.evaluate(model,0,0,0)))

