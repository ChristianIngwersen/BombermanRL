import torch

class EvolutionaryStrategy:

    def __init__(self, model, fitness, impact, learning_rate=0.5, transfer=False):
        self.model = model(transfer=transfer)
        self.fitness = fitness()
        self.learning_rate = learning_rate
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
        reward = self.fitness.evaluate(self.model, epsilon, self.learning_rate, self.impact, 5)
        return reward, seed

    def play_game(self, num_episode):
        reward = self.fitness.evaluate(self.model, 0, 0, 0, num_episode)
        return reward

