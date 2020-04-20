from learners import *
from simulators import *
from models import WrappedModel
import torch.nn as nn

fc_layer = lambda x, y: nn.Sequential(nn.Linear(x, y), nn.ReLU(), nn.Dropout(0.5))

model = nn.Sequential(fc_layer(4, 256), fc_layer(256, 256), fc_layer(256, 64), nn.Linear(64, 2))
model = WrappedModel(model)
simulator = GymSimulator('CartPole-v0')
learner = DeepQLearner(simulator, model)
learner.train()