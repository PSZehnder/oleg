import yaml
import os
from collections import defaultdict
import os.path as osp
import copy
from utils import printargs

CONFIGS, _ = osp.split(osp.realpath(__file__))
CONFIGS = osp.join(CONFIGS, 'default_configs')
DEFAULT_CONFIG = os.path.join(CONFIGS, 'base.yaml')

def merge(source, dest):
    out = defaultdict(dict, copy.copy(dest))
    if not isinstance(out, dict):
        return source

    if not isinstance(source, dict):
        return source

    if len(source) == 0:
        return out

    for k, v in source.items():
        merged = merge(v, out[k])
        out[k] = merged

    return out

class Options:

    def __init__(self, *data_list):
        DEFAULT_CONFIG = osp.join(CONFIGS, 'base.yaml')
        self.args = self._loadyaml(DEFAULT_CONFIG)
        self.args = self.loadargs(*data_list)

    def _loadyaml(self, data):

        with open(data, 'r') as yamlstream:
            out = yaml.safe_load(yamlstream)
        if out is None: #failed to load yaml
            print('failed to load %s' % data)
            out = {}
        return defaultdict(dict, out)

    def loadargs(self, *data_list):
        if len(data_list) == 0:
            return self.args
        for data in data_list:
            if isinstance(data, str):
                moreargs = self._loadyaml(data)
            elif isinstance(data, dict):
                moreargs = data
            return merge(moreargs, self.args)

    def __str__(self):
        return printargs(self.args)

class QLearnerOptions(Options):

    def __init__(self, *data_list):
        super(QLearnerOptions, self).__init__()
        DEFAULT_CONFIG = osp.join(CONFIGS, 'qlearner.yaml')
        self.args = self.loadargs(DEFAULT_CONFIG, *data_list)

