import torch.nn as nn
import copy
import torch
from utils import *

def build_model(configfile):
    pass

class BaseModel(nn.Module):

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        torch.set_default_dtype(torch.double)
        super(BaseModel, self).__init__()

    def load_recent(self, path):
        model = mostrecent(path)
        self.load_state_dict(torch.load(model))

    def copy(self):
        new_model = copy.deepcopy(self)
        return new_model

    # cast to a proper tensor
    def preprocess_state(self, state):
        if not isinstance(state, torch.Tensor):
            try:
                out = torch.tensor(state).to(self.device)
            except:
                raise TypeError('could not cast %s to tensor' % type(state))
        else:
            try:
                out = state.to(self.device)
            except:
                raise TypeError('coud not cast to %s' % self.device)
        return out

    # use an optional backwards hook to clamp gradients
    # from https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
    # call once and good to go
    def clampgrad(self, clip=1):
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

    def __call__(self, state):
        state = self.preprocess_state(state)
        output = super(BaseModel, self).__call__(state)
        return output

class WrappedModel(BaseModel):

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def __call__(self, state):
        state = self.preprocess_state(state)
        output = self.model(state)
        return output
