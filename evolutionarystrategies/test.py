from evolutionarystrategy import evolutinarystrategy
from fitness import fitness
from model import model


Model = model()
Fitness = fitness()
Evostrat = evolutinarystrategy(Model, Fitness)


#for _ in range(5):
#    Evostrat.evolution()
#    print(Model.params)

Evostrat.evolution()