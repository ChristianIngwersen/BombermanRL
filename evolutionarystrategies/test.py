from evolutionarystrategy import EvolutionaryStrategy
from fitness import Fitness
from model import Model


model = Model()
fitness = Fitness()
evo_strat = EvolutionaryStrategy(model, fitness)


for _ in range(5):
    evo_strat.evolution()
