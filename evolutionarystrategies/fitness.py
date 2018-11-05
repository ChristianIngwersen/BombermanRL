### fitness python file
#import numpy as np
import pommerman
from pommerman import agents
from pommerman_script import make_env



class fitness():

    # TODO: setup enviroment, add env to init and import pommerman
    def __init__(self):
        self.is_made = True
        self.num_episode = 1
        self.render = True

        self.env = make_env("PommeFFAPartialFast-v0")

    def max(self, model, epsilon):
        return sum(model.params)+sum(epsilon)

    def evaluate(self, model, epsilon):
        tmp_model = model.__copy__()
        for key, weights in epsilon.items():
            tmp_model.params[key] += weights

        return self.run_game(tmp_model)

    def run_game(self, model):
        # Run the episodes just like OpenAI Gym
        for i_episode in range(self.num_episode):
            state = self.env.reset()
            done = False
            fitness = 0
            while not done:
                if self.render:
                    self.env.render()

                #print(state)
                actions = model.act(state)
                state, reward, done, info = self.env.step(actions)

                fitness += self.survive_fitness(state)


            print('Episode {} finished'.format(i_episode))
            print(fitness)
        self.env.close()

        return fitness

    # Fitness function based on surviving for as long as possible
    def survive_fitness(self, playerstate):
        #print(playerstate)
        #if 10 in playerstate['alive']:
        #    return 1
        #else:
        #    return 0
        print(playerstate)
        return 1

    # TODO: define fitness function for evn