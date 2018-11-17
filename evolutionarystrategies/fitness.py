#from pommerman_script import make_env
#import cloudpickle
#import pickle
from envs import make_vec_envs
import torch

class Fitness:

    def __init__(self,individuals):
        self.num_episode = 5
        self.render = False
        self.envs = make_vec_envs("PommeFFAPartialFast-v0",1, individuals, 0.99, False, 1,
        '/home/jakob/02456/BombermanRL/evolutionarystrategies/tmp/gym/', False, torch.device("cpu"), allow_early_resets=False)
        #self.env = cloudpickle.dumps(make_env("PommeFFAPartialFast-v0"))
        #if individuals==1:
        #    self.env = make_env("PommeFFAPartialFast-v0")
        #else:
        #    self.env = [make_env("PommeFFAPartialFast-v0") for _ in range(individuals)]
        self.train = True

    def max(self, model, epsilon):
        return sum(model.params)+sum(epsilon)

    def evaluate(self, model, epsilon, learning_rate,impact,id):
        tmp_model = model.copy()
        if epsilon==0:
        	self.train=False
        	pass
        else:
        	self.train=True
        	for key, weights in epsilon.items():
        		tmp_model.params[key] += learning_rate*weights
        return self.run_game(tmp_model,impact,id)

    def run_game(self, model,impact,id):
        # Run the episodes just like OpenAI Gym
        #self.env = pickle.loads(self.env)
        env = self.envs[id]
        fitness = []
        for i_episode in range(self.num_episode):
            state = env.reset()
            done = False
            episode_fitness = 0
            while not done:
                if self.render:
                    env.render()
                actions = model.act(state)
                state, reward, done, info = env.step(actions)
                if self.train:
                	episode_fitness += self.survive_fitness(impact,env)/1000
            if reward>0:
                imp_total = sum(sum(impact[key]) for key in impact)
                episode_fitness += reward*(1-imp_total)
            fitness.append(episode_fitness)
        self.env[id].close()
        return sum(fitness)/len(fitness)

    # Fitness function based on surviving for as long as possible
    def survive_fitness(self,impact,env):
        e
        state = env.env.get_observations()[0]
        score = 0

        if not 11 in state['alive']:
            score += 1*impact['imp_enemies'][0]
        if not 12 in state['alive']:
            score += 1*impact['imp_enemies'][1]
        if not 13 in state['alive']:
            score += 1*impact['imp_enemies'][2]
        if 10 in state['alive']:
            score += 1*impact['imp_team'][0]
        if state['can_kick']:
            score += 1*impact['imp_powerup'][0]

        return score
