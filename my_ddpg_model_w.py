# Implementation based on UDACITY DRLND course

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_QNetwork(nn.Module):
    """Actor (Policy) Model. DuelingQNetwork"""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=200,  init_weights=3e-3):
        """Initialize parameters and build model.
        Normalization of Layers improves network perofmance significantly.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor_QNetwork, self).__init__()
        # Random seed
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        # Input layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # create hidden layers according to HIDDEN_SIZES
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters(init_weights)

        # Initialize Parameter
    def reset_parameters(self, init_weights):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_weights, init_weights)


    def forward(self, state):
        """Build a network that maps state -> action values. Forward propagation"""

         # classical network with relu activation function
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.tanh(x)


class Critic_QNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=200, init_weights=3e-3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic_QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Input Layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Hidden Layer
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters(init_weights)

        # Initialize Parameter
    def reset_parameters(self, init_weights):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_weights, init_weights)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
