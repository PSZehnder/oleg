import torch.nn as nn
import copy
import torch
from utils import *

def build_model(configfile):
    pass

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def load_recent(self, path):
        model = mostrecent(path)
        self.load_state_dict(torch.load(model))

    def copy(self):
        new_model = copy.deepcopy(self)
        return new_model.load_state_dict(self.state_dict())

    def choose_action(self, state):
        raise NotImplementedError

