### evolutionarystrategy python file
#import random
import torch

class evolutinarystrategy():

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
            reward = self.fitness.evaluate(self.model, epsilon)

            # book keeping
            epsilons.append(epsilon)
            rewards.append(reward)

        # update the parameters of the model, based on the reward and epsilon
        self.model.updateparams(epsilons, rewards, self.learning_rate)