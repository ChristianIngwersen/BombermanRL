from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model
import sys

evo_strat = EvolutionaryStrategy(Model, Fitness, processes=1, learning_rate = 5e-4)

impact = {
'imp_team': [0.2],
'imp_enemies': [0.2,0.2,0.2],
'imp_powerup': [0.2]
}

for i in range(5):
    evo_strat.evolution(impact, 0)
    if (i-1)%100==0:
    	print("Average win rate over 5 games {}".format(evo_strat.play_game(impact)))
