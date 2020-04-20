import torch
from options import *
from utils import *
import random
from collections import namedtuple
from .visdom import *

class Learner:

    def __init__(self, simualtor, model, options=None):

        if options is None:
            self.options = Options().args
        else:
            self.options = options

        printargs(self.options)

        self.model = model

        self.name = self.options['name']

        # set up optimizer
        optimizer_args = self.options['optimizer']
        self.max_lr = optimizer_args['learning_rate']['max_lr']
        self.min_lr = optimizer_args['learning_rate']['min_lr']
        self._initoptim(optimizer_args)

        # set up training args
        training_args = self.options['training']
        self.num_epochs = training_args['epochs']
        self.loss_func = training_args['loss']
        self.batch_size = training_args['batch_size']

        self.memory = ReplayBuffer(training_args['memory_len'])
        self.explore_val = training_args['max_epsilon']
        self.min_explore_val = training_args['min_epsilon']
        self.explore_decay = training_args['epsilon_decay']
        self.gamma = training_args['gamma']

        # detect device
        if training_args['device'] is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(training_args['device'])

        self.model = self.model.to(self.device)

        # io considerations; frequencies are in epochs
        io_args = self.options['io']
        self.render_path = io_args['render']['render_path']
        self.render_frequency = io_args['render']['render_frequency']
        self.save_frequency = io_args['save']['save_frequency']
        self.weights_path = io_args['save']['save_path']

        self.avg_window = io_args['avg_window']

        if self.render_path is None:
            self.render_path = self.name + '_latest/render'
        if self.weights_path is None:
            self.weights_path = self.name + '_latest/weights'

        for path in (self.weights_path, self.render_path):
            os.makedirs(path, exist_ok=True)

        # load simulator
        self.simulator = simualtor
        self.simulator.reset()

        # store visdom options. launch visdom server
        vis_args = self.options['visdom']
        self.port = vis_args['port']
        self.use_visdom = vis_args['use_visdom']
        self.visdom_avg = vis_args['avg']
        self.visdom_var = vis_args['var']
        try:
            launch_visdom(self.use_visdom, self.port)
        except:
            print('failed to launch visdom. Try again with: \npython -m visdom.server -port %d' % self.port)
        # set up logger
        self.logger = Logger(['loss', 'reward', 'game_len'], avg=self.visdom_avg, var=self.visdom_var)

    def update_scheduler(self, loss=None):
        if self.scheduler_name == 'plateau':
            self.scheduler.step(loss)
        else:
            self.scheduler.step()


    def _initoptim(self, argsdict):
        if isinstance(argsdict['args'], dict):
            self.optimizer_name, self.optimizer = getoptimizer(self.model.parameters(), lr=self.max_lr, name=argsdict['name'],
                                                               **argsdict['args'])
        else:
            self.optimizer_name, self.optimizer = getoptimizer(self.model.parameters(), lr=self.max_lr, name=argsdict['name'])
        schedule_args = argsdict['learning_rate']['scheduler']
        self.scheduler_name, self.scheduler = getscheduler(self.optimizer)

    def render(self, epoch):
        raise NotImplementedError

    def clampgrad(self, params):
        for param in params:
            param.grad.data.clamp_(-1, 1)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# taken from pytorch example code
class ReplayBuffer:

    def __init__(self, len):
        self.capacity = len
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)