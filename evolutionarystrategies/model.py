### model python file
import numpy as np

class model():
    # TODO: change model to be a neural network
    def __init__(self):
        self.params = [0]*10
        self.reward = -1

    # TODO: change update params to allow for updateing of neural network.
    def updateparams(self, epsilon, rewards, learningrate):
        best_reward = np.argmax(rewards)
        self.params = self.params + epsilon[best_reward]

    def shape(self):
        return len(self.params)

    # TODO: function to mimic agents and to take actions. Look at the code from the guy.
    def act(self, state):
        return 0