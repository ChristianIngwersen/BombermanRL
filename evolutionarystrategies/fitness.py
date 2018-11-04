### fitness python file
#import numpy as np
class fitness():

    # TODO: setup enviroment, add env to init and import pommerman
    def __init__(self):
        self.is_made = True

    def max(self, model, epsilon):
        return sum(model.params)+sum(epsilon)


    # TODO: define fitness function for evn