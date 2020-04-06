from .baselearner import Learner
from options import CONFIGS
import os

class DQNLearner(Learner):

    def __init__(self, model, optionspath=os.path.join(CONFIGS, 'dqn.yaml')):
        super(DQNLearner, self).__init__(model, optionspath)