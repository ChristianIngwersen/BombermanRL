### fitness python file
#import numpy as np
class fitness():

    def __init__(self):
        self.is_made = True

    def max(self,model, epsilon):
        return sum(model.params)+sum(epsilon)
