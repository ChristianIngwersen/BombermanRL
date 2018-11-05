from pommerman_script import make_env


class Fitness:

    def __init__(self):
        self.num_episode = 5
        self.render = False
        self.env = make_env("PommeFFAPartialFast-v0")
        self.train = True

    def max(self, model, epsilon):
        return sum(model.params)+sum(epsilon)

    def evaluate(self, model, epsilon, learning_rate,impact):
        tmp_model = model.copy()
        if epsilon==0:
        	self.train=False
        	pass
        else:
        	self.train=True
        	for key, weights in epsilon.items():
        		tmp_model.params[key] += learning_rate*weights
        return self.run_game(tmp_model,impact)

    def run_game(self, model,impact):
        # Run the episodes just like OpenAI Gym
        fitness = []
        for i_episode in range(self.num_episode):
            state = self.env.reset()
            done = False
            episode_fitness = 0
            while not done:
                if self.render:
                    self.env.render()
                actions = model.act(state)
                state, reward, done, info = self.env.step(actions)
                if self.train:
                	episode_fitness += self.survive_fitness(impact)/1000
            if reward>0:
            	episode_fitness += reward*(1-impact)
            fitness.append(episode_fitness)
        self.env.close()
        return sum(fitness)/len(fitness)

    # Fitness function based on surviving for as long as possible
    def survive_fitness(self,impact):
        state = self.env.env.get_observations()[0]
        if 10 in state['alive']:
            return 1*impact
        else:
            return 0
