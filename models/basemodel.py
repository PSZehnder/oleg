import torch.nn as nn
import copy
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

    # by default is the identity function
    def preprocess_state(self, state):
        return state

    def __call__(self, state):
        state = self.preprocess_state(state)
        output = super(BaseModel, self).__call__(state)
        return output

class WrappedModel(BaseModel):

    def __init__(self, model):
        super(WrappedModel, self). __init__()
        self.model = model

    def __call__(self, state):
        state = self.preprocess_state(state)
        output = self.model(state)
        return output
