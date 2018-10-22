# BombermanRL
DTU course 2456 Deep learning project. 
We've chosen the Pommerman reinforcement learning project. The enviroment used is the [playground enviroment](https://github.com/MultiAgentLearning/playground) used for the NIPS 2018 Pommerman competion (<https://www.pommerman.com>).

## Project goal
The overall goal with the project is to make a submission to the NIPS 2018 competition (*Deadline November 21st*).

**The subgoals to our agents are the following:**
  - Train a consistent FFA agent to beat three RandomAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat one RandomAgent and two SimpleAgents on average more then 50% of the times
  - Train a consistent FFA agent to beat three SimpleAgents on average more then 50% of the times
  
 If we succed on the above goals it would be interesting to add **_imitation learning_** to our agents in order to make two agents collaborate in the competetion. 
  
## Policy gradient learning vs. evolutionary learning
We'll start by using [Ross Wightman's PyTorch model](https://github.com/rwightman/pytorch-pommerman-rl) as a starting point. He succeds to beat three SimpleAgents **_95%_** of the games using policy gradient learning. 

We believe that evolutionary learning will be a good approach to the Pommerman problem because the reward of the game is only calculated once, the variance of each action evaluation is less impactful and since the evaluation time is far smaller, more training steps can be achieved.
