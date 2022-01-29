import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb


class PolicyNet(nn.Module):
    def __init__(self, input_feat=55):
        super(PolicyNet, self).__init__()

        # self.fc1 = nn.Linear(55, 64)
        # self.fc2 = nn.Linear(64, 36)
        # self.fc3 = nn.Linear(36, 72)
        # self.fc4 = nn.Linear(72,36)
        # self.fc5 = nn.Linear(36,18)
        # self.fc6 = nn.Linear(18, 5)
        self.cls = nn.Sequential(
        nn.Linear(in_features=input_feat, out_features=512, bias=True),
        # nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Linear(in_features=512, out_features=128, bias=True),
        # nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64, bias=True),
        # nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=16, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=5, bias=True)
        )
    def forward(self, x):
        # x = nn.Relu
        x = self.cls(x)
        # print(x)
        x = F.softmax(x, dim=-1)
        return x