# DRLND---Collaboration
## Third project in the Deep Reinforcement Learning Nanodegree.

![Tennis.](tennis.png)

# Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

    After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Installation
To set up your python environment to run the code in this repository, follow the instructions below.
Create (and activate) a new environment with Python 3.6.
 Linux or Mac:
```
    conda create --name drlnd python=3.6
    source activate drlnd
```
  Windows:
```
    conda create --name drlnd python=3.6 
    activate drlnd
```
Clone the following repository , and navigate to the python/ folder. Then, install several dependencies.
```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
```
Create an IPython kernel for the drlnd environment.
```
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
More informnation can be found here:
https://github.com/udacity/deep-reinforcement-learning#dependencies.

The environment can be downloaded using the following links:


    Linux:            https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
    Mac OSX:          https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip



The environment is build using Unity ML-agent. More details about these environments and how to get started can be found here:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md

# Train the Agent
Finally you just need to run `MADDPG_Tennis.ipynb` to load packages, start the environment and train the agent. You may need to update the environment path in dependency of Operating System.
