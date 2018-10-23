# BombermanRL
DTU course 2456 Deep learning project. 
We've chosen the Pommerman reinforcement learning project. The environment used is the [playground enviroment](https://github.com/MultiAgentLearning/playground) used for the NIPS 2018 Pommerman competition (<https://www.pommerman.com>).

## Motivation
Reinforcement learning is still a field in rapid development. A currently on-going competition on NIPS is to explore Multi-agent Reinforcement learning. We want to participiate in this current on-going exploration and study different methodics and their performance for this given problem. 

Further motivation for why multi-agent learning is interesting can be seen in their explainer page;
'Accomplishing tasks with infinitely meaningful variation is common in the real world and difficult to simulate. Competitive multi-agent learning enables this.'

## Background
We'll start by using [Ross Wightman's PyTorch model](https://github.com/rwightman/pytorch-pommerman-rl) as a starting point. With his model he succeeds to beat three SimpleAgents **_95%_** of the games using policy gradient learning.

We believe that evolutionary learning will be a good approach to the Pommerman problem. Policy gradient is usually better than evolutionary learning if the actual reward is calculated each time an action is taken. This is because the variance of the evaluation of evolutionary learning per action is high. This environment only calculates the expected reward per action and the actual reward is only calculated when the game ends. Thus the variance of the expected reward is less impactful of the actual reward. The evaluation time of evolutionary learning is a lot lower than policy gradiant. Therefore, more training evaluations can be done and presumably a higher performance can be achieved in the same time from evolutionary learning.

To test our hypothesis we'll try to train an agent using evolutionary learning and compare our results to the results obtained by *Ross Wightman*.
 

## Milestones
The overall goal with the project is to make a submission to the NIPS 2018 competition (*Deadline November 21st*).

**The subgoals to our agents are the following:**
  - Train a consistent FFA agent to beat three RandomAgents on average more than 50% of the times
  - Train a consistent FFA agent to beat one RandomAgent and two SimpleAgents on average more than 50% of the times
  - Train a consistent FFA agent to beat three SimpleAgents on average more than 50% of the times

Here the **_RandomAgents_** are agents taking completely random actions and **_SimpleAgents_** are benchmark agents given by the Pommerman community as a benchmark on how good an agent should be before a submission. 

When we have a succesfull agent for the FFA enviroment, we'll expand it to the Team environment, which is the official NIPS 2018 Competition environment.

Furthermore if we succeed on the subgoals it would be interesting to add **_imitation learning_** to our agents in order to make two agents collaborate while still following the rules of the NIPS 2018 competetion.
 





