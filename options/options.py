import yaml
import os
from collections import defaultdict
import warnings
import torch.nn as nn
import torch.optim as optim

CONFIGS = os.path.realpath(__file__)
DEFAULT_CONFIG = os.path.join(CONFIGS, 'base.yaml')

def makedefdict(data):
    return defaultdict(makedefdict({}), data)

def merge(source, dest):
    if len(source) == 0:
        return dest

    for k, v in source.items():
        dest['k'] = merge(v, dest['k'])

class Options:

    def __init__(self, yamlpath):
        self.args = self._loadyaml(DEFAULT_CONFIG)
        user_args = self._loadyaml(yamlpath)

        self.args = merge(self.args, user_args)

    def _loadyaml(self, yamlpath):
        with open(yamlpath, 'r') as yamlstream:
            data = yaml.safe_load(yamlstream)
        return makedefdict(data)

    def loadargs(self, yamlpath):
        moreargs = self._loadyaml(yamlpath)
        self.args = merge(self.args, moreargs)

def _tryoptim(params, lr, optfunc, **kwargs):
    try:
        return optfunc(params, lr=lr, **kwargs)
    except TypeError:
        warnings.warn('failed to load optimizer args, using defaults')
        return optfunc(params, lr=lr)

def getoptimizer(params, lr, name=None, **kwargs):
    if name is None:
        name = 'Adam'
    if name.lower() == 'adadelta':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.Adadelta, **kwargs)
    if name.lower() == 'adam':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.Adam, **kwargs)
    if name.lower() == 'adagrad':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.Adagrad, **kwargs)
    if name.lower() == 'adamw':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.AdamW, **kwargs)
    if name.lower() == 'adamax':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.Adamax, **kwargs)
    if name.lower() == 'lbfgs':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.LBFGS, **kwargs)
    if name.lower() == 'rmsprop':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.RMSProp, **kwargs)
    if name.lower() == 'sgd':
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.SGD, **kwargs)
    return name, optimizer

def getscheduler(optimizer, name=None, **kwargs):
    if name is None:
        return None, None
    if name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name.lower() == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    if name.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    if name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if name.lower() == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    return name, scheduler

def getloss(name):
    if name is None:
        name = 'mse'
    if name.lower() == 'mse':
        loss = nn.MSELoss()
    if name.lower() == 'l1':
        loss = nn.L1Loss()
    if name.lower() == 'smoothl1':
        loss = nn.SmoothL1Loss()
    return loss

