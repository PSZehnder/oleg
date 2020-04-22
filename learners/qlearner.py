from .baselearner import Learner, Transition
from options import *
from utils import trans2tens
import os.path as osp
import torch
from .visdom import VisdomDictPlotter, VisdomVideoPlotter
import time
import random

class DeepQLearner(Learner):

    def __init__(self, simulator, model, optionspath=None):
        options = QLearnerOptions(optionspath).args
        super(DeepQLearner, self).__init__(simulator, model, options)

        target_options = self.options['training']
        self.target_options = target_options

        self.batches = target_options['batches']
        self.episode_update = target_options['episode_update']
        self.target_update = target_options['target_update']
        self.target_model = None
        if self.target_update == 0:
            self.target_model = self.model

        self.update_target(0)

        if self.use_visdom:
            global plotter
            plotter = VisdomDictPlotter(env_name=self.name)

            global simplotter
            simplotter = VisdomVideoPlotter(env_name=self.name) # DIS KINDA BROKE

    def update_target(self, epoch):
        if self.target_update <= 0:
            return

        elif self.target_model is None:
            self.target_model = self.model.copy()

        elif epoch % self.target_update == 0:
            self.target_model = self.model.copy()

    def save_model(self, epoch):
        if (epoch + 1 ) % self.save_frequency == 0:
            torch.save(self.model, osp.join(self.weights_path, '%s_epoch_%s.pth' % (self.name, epoch)))

    def train(self):
        printargs(self.target_options)
        print('----- BEGIN Q LEARNING ----- \n')

        for i in range(self.num_epochs):

            if self.explore_val >= self.min_explore_val:
                self.explore_val = self.explore_val * self.explore_decay

            # variables for logging
            num_actions = 0
            cum_reward = 0
            cum_loss = 0
            done = False
            tick = time.time()

            while not done:
                state = self.simulator.state
                state = self.simulator.preprocess(state)

                action = self.choose_action(state, self.explore_val)
                reward = self.simulator.update(action)
                cum_reward += reward
                num_actions += 1

                next_state = self.simulator.state
                next_state = self.simulator.preprocess(next_state)
                done = self.simulator.done

                trans = Transition(state, action, next_state, reward, done)

                # can update while playing like in the Nature paper
                self.memory.push(trans2tens(trans, device=self.device))
                if self.episode_update <= 1:
                    cum_loss += self.memory_replay(self.batch_size, self.batches)

            # can update after playing (this can be more efficient for GPU with lots of RAM)
            if self.episode_update >= 1:
                if (i + 1) % self.episode_update == 0:
                    cum_loss += self.memory_replay(self.batch_size, self.batches)
                    cum_loss = cum_loss / self.batches
            else:
                cum_loss = cum_loss / num_actions

            if (i + 1) % self.render_frequency == 0:
                self.simulator.render(self.model, os.path.join(self.render_path, 'epoch_%s' % i))
                if self.use_visdom:
                    simplotter.update(self.render_path)

            self.update_target(i)
            self.simulator.reset()

            # update our logs
            self.logger.update(cum_loss, cum_reward, num_actions)
            if self.use_visdom:
                plotter.plot(self.logger.asdict())
            print(str(self.logger) + 'epsilon: %.2f' % self.explore_val + ' time: %.2f' % (time.time() - tick))

            if 'avg' in self.logger['reward']:
                if len(self.logger['reward']['avg']):
                    self.update_scheduler(self.logger['reward']['avg'][-1]) # the plateeau scheduler needs a validation score
                    if self.logger['reward']['avg'][-1] >= 150:
                        self.simulator.render(self.model, self.render_path)
                        print('completed because good score!')
                        break

    def choose_action(self, state, explore_val):
        num = random.uniform(0, 1)
        if num > explore_val:
            action = self.model(state)
            action = torch.argmax(action).detach().cpu().item()
        else:
            action = self.simulator.rand_action()
        return action

    # from pytorch documentation
    def memory_replay(self, batch_size, num_batches):
        if len(self.memory) < 8:
            return 0

        cumulative_loss = 0

        for b in range(num_batches):
            size = min(len(self.memory), batch_size)
            batch = self.memory.sample(size)
            batch = Transition(*zip(*batch))

            nonfinal_next = [batch.next_state[i]for i, done in enumerate(batch.done) if not done]
            nonfinal_next = torch.stack(nonfinal_next)

            state = torch.stack(batch.state)
            actions = torch.stack(batch.action)
            rewards = torch.stack(batch.reward)
            done = torch.stack(batch.done)

            # get score with policy network
            modeled = self.model(state)
            experienced_values = torch.gather(modeled, 1, actions.unsqueeze(1))

            # get score with target network
            target_values = torch.zeros(size, device=self.device)
            target_values[~ done] = torch.max(self.target_model(nonfinal_next), dim=1).values.float()
            target_values = (target_values * self.gamma) + rewards

            # compute loss
            loss = self.loss_func(experienced_values, target_values.unsqueeze(1))

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.clampgrad(self.model.parameters())
            self.optimizer.step()

            cumulative_loss += loss.item()

        return cumulative_loss




