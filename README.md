# BombermanRL
DTU course 2456 Deep learning project. 
We've chosen the Pommerman reinforcement learning project. The enviroment used is the [playground enviroment](https://github.com/MultiAgentLearning/playground) used for the NIPS 2018 Pommerman competion (<https://www.pommerman.com>).

## Project goals
The overall goal with the project is to make a submission to the NIPS 2018 competition (*Deadline November 21st*).

**The subgoals to our agents are the following:**
  - Train a consistent FFA agent to beat three RandomAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat one RandomAgent and two SimpleAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat three SimpleAgents on average more then 50% of the times

Here the **_RandomAgents_** are agents taking completely random actions and **_SimpleAgents_** is a benchmark agent given by the Pommerman community as a benchmark on how good an agent should be before a submission. 

When we have a succesfull agent for the FFA enviroment, we'll expand it to the Team enviroment which is the official NIPS 2018 Competition environment.
  
## Approach
We'll start by using [Ross Wightman's PyTorch model](https://github.com/rwightman/pytorch-pommerman-rl) as a starting point. With his model he succeds to beat three SimpleAgents **_95%_** of the games using policy gradient learning. 

#### Policy gradient learning vs. evolutionary learning
We believe that evolutionary learning will be a good approach to the Pommerman problem. Policy gradient is usually better than evolutionary learning if the actual reward is calculated each time an action is taken since the variance of the evaluation of evolutionary learning per action is high. This environment only calculates the expected reward per action and the actual reward is only calculated when the game ends. Thus the variance of the expected reward is less impactful of the actual reward. The evaluation time of evolutionary learning is a lot lower than policy gradiant. Therefore, more training evaluations can be done and presumably a higher performance can be achieved in the same time from evolutionary learning.

To test our hypothesis we'll try to train an agent using evolutionary learning and compare our results to the results obtained by *Ross Wightman*.

Furthermore if we succed on the subgoals it would be interesting to add **_imitation learning_** to our agents in order to make two agents collaborate while still following the rules of the NIPS 2018 competetion. 




