import torch
import numpy as np
import pathos.multiprocessing as mp


class EvolutionaryStrategy:

    def __init__(self, model, fitness, processes=4, populationsize=10, learning_rate=0.5):
        self.model = model()
        self.processes = processes
        self.fitness = fitness(individuals=populationsize)
        self.learning_rate = learning_rate
        self.populationsize = populationsize

    def evolution(self,id):
        
        epsilon = {}
        for key, shape in self.model.shape().items():
            if self.model.params[key].type() == "torch.FloatTensor":
                epsilon[key] = torch.randn(shape).float()
            elif self.model.params[key].type() == "torch.LongTensor":
                epsilon[key] = torch.randn(shape).long()
            else:
                epsilon[key] = torch.randn(shape)
        # fitness function
        reward = self.fitness.evaluate(self.model, epsilon,self.learning_rate, 0.5,id)
        return reward, epsilon    # book keeping
        
        

    def play_game(self):
        reward = self.fitness.evaluate(self.model, 0, 0, 0,1)

        return reward

    def parallel_evolution(self):

        pool = mp.Pool(processes=self.processes)
        results = pool.map(self.evolution,range(self.populationsize))
        rewards = [r[0] for r in results]
        epsilons = [r[1] for r in results]
        # Setup a list of processes that we want to run
        #processes = [mp.Process(target=self.evolution, args=(x, output)) for x in range(permutations)]
        self.model.update_params(epsilons, rewards, self.learning_rate)
        # Run processes
        '''for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]
        '''
        #print(results)
    def sequential_evolution(self):
        epsilons = []
        rewards = []
        for _ in range(self.populationsize):
            reward, epsilon = self.evolution(1)
            epsilons.append(epsilon)
            rewards.append(reward)
        self.model.update_params(epsilons, rewards, self.learning_rate)

