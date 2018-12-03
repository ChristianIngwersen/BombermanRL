#from envs.pommerman import make_env
from envs import make_env
import numpy as np



class Fitness:

    def __init__(self):
        self.render = False
        self.id = np.random.randint(0, 100)
        #self.env = make_env("PommeFFAPartialFast-v0")
        self.envs = make_env("PommeFFAPartialFast-v0", 1, 1, './tmp/gym/', False, False)
        self.env = self.envs().env
        self.train = True

    def evaluate(self, model, epsilon, learning_rate, impact, num_episode):
        if epsilon == 0:
            self.train = False
            return self.run_game(model, 0, num_episode)
        else:
            self.train = True
            for key, weights in epsilon.items():
                model.params[key] += learning_rate * weights
            return self.run_game(model, impact, num_episode)

    def run_game(self, model, impact, num_episode):
        # Run the episodes just like OpenAI Gym
        fitness = []
        for i_episode in range(num_episode):
            game_length = 0
            state = self.env.reset()
            done = False
            episode_fitness = 0
            while not done:
                if self.render:
                    self.env.render()
                actions = model.act(state)
                state, reward, done, info = self.env.step(actions)
                game_length += 1
                if self.train:
                    episode_fitness += self.survive_fitness(impact, self.env)
            episode_fitness /= game_length
            if reward > 0:
                if not impact == 0:
                    imp_total = sum(sum(impact[key]) for key in impact)
                    episode_fitness += reward * (1 - imp_total)
                else:
                    episode_fitness += reward
            fitness.append(episode_fitness)
            self.env.close()
        return sum(fitness) / len(fitness)

    # Fitness function based on surviving for as long as possible
    def survive_fitness(self, impact, env):
        if impact == 0:
            return 0
        state = env.env.get_observations()[0]
        score = 0
        if not 11 in state['alive']:
            score += 1 * impact['imp_enemies'][0]
        if not 12 in state['alive']:
            score += 1 * impact['imp_enemies'][1]
        if not 13 in state['alive']:
            score += 1 * impact['imp_enemies'][2]
        if 10 in state['alive']:
            score += 1 * impact['imp_team'][0]
        if state['can_kick']:
            score += 1 * impact['imp_powerup'][0]

        return score
