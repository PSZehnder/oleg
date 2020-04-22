import warnings
import torch.optim as optim
import torch.nn as nn
import os
import torch
from statistics import variance
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# convert strings to their pytorch objects
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
        optimizer = _tryoptim(params, lr=lr, optfunc=optim.RMSprop, **kwargs)
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
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
    else:
        return  nn.MSELoss()
    return loss

def printargs(args, tabs=4, depth=0):
    # print nested args in a pretty way
    def _buildargs(args, tabs, depth):
        out_str = ''
        for k, v in args.items():
            if not isinstance(v, dict):
                out_str += '\t' * depth + '%s: ' % k + str(v) + '\n'
            else:
                out_str += '\t' * depth + '%s: ' % k + '\n'
                out_str += _buildargs(v, tabs, depth=depth + 1)
        return out_str.expandtabs(tabs)

    print(_buildargs(args, tabs, depth))

# get most recent file from folder
def mostrecent(folder):
    files = os.listdir(folder)
    files.sort(key=os.path.getmtime)
    return files[0]

# turn Transition to tensor
def trans2tens(transition, device=torch.device('cpu')):
    out = []
    for item in transition:
        if not isinstance(item, torch.Tensor):
            out.append(torch.tensor(item, device=device))
        else:
            out.append(item.to(device))
    return Transition(*out)

# take rolling average with sliding window. returns list of same shape
def rollingaverage(lst, window=100):
    out = lst.copy()
    for i, elem in enumerate(lst):
        if i < window:
            num = sum(lst[:i])
            out[i] = num / (i + 1)
        else:
            num = sum(lst[i - window: i])
            out[i] = num / window
    return out

def rollingvariance(lst, window=100):
    out = lst.copy()
    for i, elem in enumerate(lst):
        if i < window:
            num = variance(lst[:i])
            out[i] = num / (i + 1)
        else:
            num = variance(lst[i - window: i])
            out[i] = num / window
    return out

# build a logging object
class Logger:

    def __init__(self, names, window=100, avg=True, var=False):
        self._keyorder = names
        self.do_avg = avg
        self.do_var = var
        self.window = window

        self.data = {}

        for name in names:
            self.data[name] = {name : []}
            if self.do_avg:
                self.data[name]['avg'] = []
            if self.do_var:
                self.data[name]['var'] = []

    def update(self, *elems):
        for i, name in enumerate(self._keyorder):
            self.data[name][name].append(elems[i])
            if self.do_avg and len(self.data[name][name]) >= 3:
                self.data[name]['avg'] = rollingaverage(self.data[name][name], self.window)
            if self.do_var and len(self.data[name][name]) >= 3:
                self.data[name]['var'] = rollingvariance(self.data[name][name], self.window)

    def __getitem__(self, item):
        return self.data[item]

    def asdict(self):
        return self.data

    def __str__(self):
        epoch = len(self.data[self._keyorder[0]][self._keyorder[0]])
        out_str = 'epoch: %s ' % epoch
        for name in self._keyorder:
            out_str += '%s: %.3f ' %(name, self.data[name][name][-1])
        return out_str