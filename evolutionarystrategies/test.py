from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys



impact = {
'imp_team': [0.2],
'imp_enemies': [0.2,0.2,0.2],
'imp_powerup': [0.2]
}

evo_strat = EvolutionaryStrategy(Model, Fitness, impact,  processes=1, populationsize=1, learning_rate = 5e-4)

for i in range(5):
    evo_strat.parallel_evolution()
    #if (i-1)%100==0:
    	#print("Average win rate over 5 games {}".format(evo_strat.play_game()))
