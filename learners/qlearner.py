from .baselearner import Learner, Transition
from options import *
from utils import trans2tens
import os.path as osp
import torch
from .visdom import VisdomDictPlotter
import time

class DeepQLearner(Learner):

    def __init__(self, simulator, model, optionspath=None):
        options = QLearnerOptions(optionspath)
        super(DeepQLearner, self).__init__(simulator, model, options)

        target_options = self.options['training']

        self.batches = target_options['batches']
        self.episode_update = target_options['episode_update']
        self.target_update = target_options['target_update']

        self.target_model = None
        self.update_target(0)

        if self.use_visdom:
            global plotter
            plotter = VisdomDictPlotter(env_name=self.name)


    def update_target(self, epoch):
        if self.target_update <= 0:
            self.target_network = self.model

        elif self.target_model is None:
            self.target_model = self.model.copy()

        elif epoch % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, epoch):
        if (epoch + 1 ) % self.save_frequency == 0:
            torch.save(self.model, osp.join(self.weights_path, '%s_epoch_%s.pth' % (self.name, epoch)))

    def train(self):
        print('----- BEGIN Q LEARNING ----- \n')
        printargs(self.options)

        for i in range(self.num_epochs):
            # variables for logging
            num_actions = 0
            cum_reward = 0
            cum_loss = 0
            done = False
            tick = time.time()

            while not done:
                state = self.simulator.state
                state = self.simulator.preprocess(state)

                action = self.model.choose_action(state)
                reward = self.simulator.update(action)
                cum_reward += reward
                num_actions += 1

                next_state = self.simulator.state
                next_state = self.simulator.preprocess(next_state)
                done = self.simulator.done

                trans = Transition(state, action, next_state, reward, done)

                # can update while playing like in the Nature paper
                self.memory.push(trans2tens(trans))
                if self.episode_update <= 1:
                    cum_loss += self.memory_replay()

            # can update after playing
            if self.episode_update > 0:
                if (i + 1) % self.episode_update:
                    cum_loss += self.memory_replay()
                    cum_loss = cum_loss / self.batches
            else:
                cum_loss = cum_loss / num_actions

            self.update_target(i)

            # update our logs
            self.logger.update(cum_loss, cum_reward, num_actions)
            self.update_scheduler(self.logger['reward']['avg'][-1])

            if self.use_visdom:
                plotter.plot(self.logger.asdict())

            print(self.logger)

    def memory_replay(self):
        pass

