import torch
import numpy as np
import multiprocessing as mp

class EvolutionaryStrategy:

    def __init__(self, model, fitness, populationsize=10, learning_rate=0.5):
        self.model = model
        self.fitness = fitness
        self.learning_rate = learning_rate
        self.populationsize = populationsize

    def evolution(self):
        rewards = []
        epsilons = []
        for i in range(self.populationsize):
            # random noise
            epsilon = {}
            for key, shape in self.model.shape().items():
                if self.model.params[key].type() == "torch.FloatTensor":
                    epsilon[key] = torch.randn(shape).float()
                elif self.model.params[key].type() == "torch.LongTensor":
                    epsilon[key] = torch.randn(shape).long()
                else:
                    epsilon[key] = torch.randn(shape)

            # fitness function
            reward = self.fitness.evaluate(self.model, epsilon, self.learning_rate, 0.5)

            # book keeping
            epsilons.append(epsilon)
            rewards.append(reward)

        # update the parameters of the model, based on the reward and epsilon
        #print('Avg reward {}'.format(np.mean(rewards)))
        self.model.update_params(epsilons, rewards, self.learning_rate)

    def play_game(self):
        rewards = []
        epsilons = []
        for i in range(self.populationsize):
            # random noise
            epsilon = {}
            for key, shape in self.model.shape().items():
                if self.model.params[key].type() == "torch.FloatTensor":
                    epsilon[key] = torch.randn(shape).float()
                elif self.model.params[key].type() == "torch.LongTensor":
                    epsilon[key] = torch.randn(shape).long()
                else:
                    epsilon[key] = torch.randn(shape)

            # fitness function
            reward = self.fitness.evaluate(self.model, epsilon, self.learning_rate, 0.5)

        return reward

    def parallel_evolution(self):
        output = mp.Queue()

        # Setup a list of processes that we want to run
        permutations = 5
        processes = [mp.Process(target=self.evolution, args=(x, output)) for x in range(permutations)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]

        print(results)


