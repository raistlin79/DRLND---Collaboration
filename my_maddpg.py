from ddpg_agent import Agent

import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.97            # discount factor
#GAMMA = 0.97           # discount factor 2539
TAU = 1e-3              # for soft update of target parameters
#tau=0.02
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic 2539
WEIGHT_DECAY = 0        # L2 weight decay



class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        self.discount_factor = GAMMA
        self.tau = TAU
        self.iter = 0

        self.maddpg_agents = [Agent(state_size, action_size, random_seed) for i in range(num_agents)]


    def get_Agent(self, agent):
        return self.maddpg_agents[agent]


    def m_act(self, states, noise):
        actions = np.array([self.maddpg_agents[i].act(states[i],noise) for i in range(self.num_agents)])
        return actions


    def m_step(self, time_step, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # DDPG Agent step
        for i in range (self.num_agents):
            self.maddpg_agents[i].step(time_step, states[i], actions[i], rewards[i], next_states[i], dones[i])


    def reset(self):
        for i in range (self.num_agents):
            # DDPG Agent reset
            self.maddpg_agents[i].reset()
