import torch
import numpy as np
#import pathos.multiprocessing as mp
import multiprocessing as mp

class EvolutionaryStrategy:

    def __init__(self, model, fitness, impact, processes=4, populationsize=10, learning_rate=0.5):
        self.model = model(transfer=False)
        self.processes = processes
        self.fitness = fitness()
        self.learning_rate = learning_rate
        self.populationsize = populationsize
        self.impact = impact

    def evolution(self):
        seed = int(torch.randint(0, 100000, (1,)))
        torch.manual_seed(seed)
        epsilon = {}
        for key, shape in self.model.shape().items():
            if self.model.params[key].type() == "torch.FloatTensor":
                epsilon[key] = torch.randn(shape).float()
            elif self.model.params[key].type() == "torch.LongTensor":
                epsilon[key] = torch.randn(shape).long()
            else:
                epsilon[key] = torch.randn(shape)
        # fitness function
        reward = self.fitness.evaluate(self.model, epsilon, self.learning_rate, self.impact, 5, id)
        return reward, seed

    def play_game(self,num_episode):
        reward = self.fitness.evaluate(self.model, 0, 0, 0, num_episode, 0)
        return reward

    def play_game(self, num_episode):
        reward = self.fitness.evaluate(self.model, 0, 0, 0, num_episode, 0)
        return reward

    def sequential_evolution(self):
        epsilons = []
        rewards = []
        for _ in range(self.populationsize):
            reward, epsilon = self.evolution(impact, 1)
            epsilons.append(epsilon)
            rewards.append(reward)
        self.model.update_params(epsilons, rewards, self.learning_rate)

