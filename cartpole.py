from learners import *
from simulators import *
from models import WrappedModel
import torch.nn as nn
from torch import load

torch.set_default_dtype(torch.float32)

# build the model and wrap it with the custom model interface
fc_layer = lambda x, y: nn.Sequential(nn.Linear(x, y), nn.ReLU())
model = nn.Sequential(fc_layer(4, 24), fc_layer(24, 16), nn.Linear(16, 2))
model = torch.load('cartpole_latest/weights/cartpole_epoch_499.pth')
model.clampgrad()

# solve the simulator
simulator = GymSimulator('CartPole-v0')
learner = DeepQLearner(simulator, model, optionspath='examples/cartpole/cartpole_finetune.yaml')
learner.train()

# launch visdom in a separate terminal with
# python -m visdom.server -p 8097