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

    def __init__(self, options_paths=DEFAULT_CONFIG):
        args = []
        if isinstance(options_paths, str):
            options_paths = [options_paths]
        for path in options_paths:
            args.append(self._loadyaml(path))
        source = args[0]
        if len(args) > 1:
            for arg in args[1:]:
                source = merge(source, arg)
        self.args = source

    def _loadyaml(self, data):

        with open(data, 'r') as yamlstream:
            out = yaml.safe_load(yamlstream)
        if out is None: #failed to load yaml
            print('failed to load %s' % data)
            out = {}
        return defaultdict(dict, out)

    def __str__(self):
        return printargs(self.args)

class QLearnerOptions(Options):

    def __init__(self, options_paths=None):
        DEFAULT_CONFIG = os.path.join(CONFIGS, 'base.yaml')
        if options_paths is None:
            paths = [DEFAULT_CONFIG, osp.join(CONFIGS, 'qlearner.yaml')]
        else:
            paths = options_paths
        super(QLearnerOptions, self).__init__(paths)

