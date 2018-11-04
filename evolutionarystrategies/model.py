### model python file
import numpy as np

class model():
    def __init__(self):
        self.params = [0]*10
        self.reward = -1


    def updateparams(self, epsilon, rewards, learningrate):
        best_reward = np.argmax(rewards)
        self.params = self.params + epsilon[best_reward]

    def shape(self):
        return len(self.params)