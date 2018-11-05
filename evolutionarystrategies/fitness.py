### fitness python file
#import numpy as np
import pommerman
from pommerman import agents


class fitness():

    # TODO: setup enviroment, add env to init and import pommerman
    def __init__(self):
        self.is_made = True
        self.num_episode = 1
        self.render = False

        # Create a set of agents (exactly four)
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            # agents.RandomAgent(),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        # Make the "Free-For-All" environment using the agent list
        self.env = pommerman.make('PommeFFACompetition-v0', agent_list)



    def max(self, model, epsilon):
        return sum(model.params)+sum(epsilon)

    def run_game(self):
        # Run the episodes just like OpenAI Gym
        for i_episode in range(self.num_episode):
            state = self.env.reset()
            done = False
            fitness = 0
            while not done:
                if self.render:
                    self.env.render()
                actions = self.env.act(state)
                state, reward, done, info = self.env.step(actions)


                playerstate = (state[0])
                fitness += self.survive_fitness(playerstate)


            print('Episode {} finished'.format(i_episode))
            print(fitness)
        self.env.close()

        return fitness

    # Fitness function based on surviving for as long as possible
    def survive_fitness(self, playerstate):
        if 10 in playerstate['alive']:
            return 1
        else:
            return 0


    # TODO: define fitness function for evn