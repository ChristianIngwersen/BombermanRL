### fitness python file
#import numpy as np
import pommerman
from pommerman import agents


class fitness():

    # TODO: setup enviroment, add env to init and import pommerman
    def __init__(self):
        self.is_made = True

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
        for i_episode in range(1):
            state = self.env.reset()
            done = False
            while not done:
                if self.render:
                    self.env.render()
                actions = self.env.act(state)
                state, reward, done, info = self.env.step(actions)
            print('Episode {} finished'.format(i_episode))
        self.env.close()


    # TODO: define fitness function for evn