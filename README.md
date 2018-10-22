# BombermanRL
DTU course 2456 Deep learning project. 
We've chosen the Pommerman reinforcement learning project. The enviroment used is the [playground enviroment](https://github.com/MultiAgentLearning/playground) used for the NIPS 2018 Pommerman competion (<https://www.pommerman.com>).

## Project goals
The overall goal with the project is to make a submission to the NIPS 2018 competition (*Deadline November 21st*).

**The subgoals to our agents are the following:**
  - Train a consistent FFA agent to beat three RandomAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat one RandomAgent and two SimpleAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat three SimpleAgents on average more then 50% of the times
  
## Approach
We'll start by using [Ross Wightman's PyTorch model](https://github.com/rwightman/pytorch-pommerman-rl) as a starting point. With his model he succeds to beat three SimpleAgents **_95%_** of the games using policy gradient learning. 

#### Policy gradient learning vs. evolutionary learning
We believe that evolutionary learning will be a good approach to the Pommerman problem because the reward of the game is only calculated once, the variance of the reward per action evaluation is less impactful and since the evaluation time is far smaller, more training steps can be achieved and thus a presumably a better performance. Policy gradient is usually better than evolutionary learning if the reward is calculated each time an action is taken.

To test our hypothesis we'll try to train an agent using evolutionary learning and compare our results to the results obtained by *Ross Wightman*.

Furthermore if we succed on the subgoals it would be interesting to add **_imitation learning_** to our agents in order to make two the agents collaborate while still following the rules of the NIPS 2018 competetion. 




