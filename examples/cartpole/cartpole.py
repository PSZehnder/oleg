from learners import *
from simulators import *
from models import WrappedModel
import torch.nn as nn

# build the model and wrap it with the custom model interface
fc_layer = lambda x, y: nn.Sequential(nn.Linear(x, y), nn.ReLU(), nn.Dropout(0.5))
model = nn.Sequential(fc_layer(4, 128), fc_layer(128, 64), nn.Linear(64, 2))
model = WrappedModel(model)

# solve the simulator
simulator = GymSimulator('CartPole-v0')
learner = DeepQLearner(simulator, model, optionspath='cartpole.yaml')
learner.train()

# launch visdom in a separate terminal with
# python -m visdom.server -p 8097