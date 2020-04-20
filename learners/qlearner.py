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
        self.update_target(0)

        if self.use_visdom:
            global plotter
            plotter = VisdomDictPlotter(env_name=self.name)

            global simplotter
            simplotter = VisdomVideoPlotter(env_name=self.name) # DIS KINDA BROKE

    def update_target(self, epoch):
        if self.target_update <= 0:
            self.target_model = self.model

        elif self.target_model is None:
            self.target_model = self.model.copy()

        elif epoch % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

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
            if self.episode_update > 0:
                if (i + 1) % self.episode_update == 0:
                    cum_loss += self.memory_replay(self.batch_size, self.batches)
                    cum_loss = cum_loss / self.batches
            else:
                cum_loss = cum_loss / num_actions

            if (i + 1) % self.render_frequency == 0:
                self.simulator.render(self.model, self.render_path)
                if self.use_visdom:
                    simplotter.update(self.render_path)

            self.update_target(i)

            # update our logs
            self.logger.update(cum_loss, cum_reward, num_actions)
            if self.use_visdom:
                plotter.plot(self.logger.asdict())
            print(self.logger + 'epsilon: %.2f' % self.explore_val + 'time: %.2f' % (time.time() - tick))

            self.update_scheduler(self.logger['reward']['avg'][-1])  # the plateeau scheduler needs a validation score

    def choose_action(self, state, explore_val):
        num = random.uniform(0, 1)
        if num > explore_val:
            action = self.model(state)
            action = torch.max(action).item()
        else:
            action = self.simulator.rand_action()
        return action

    # from pytorch documentation
    def memory_replay(self, batch_size, num_batches):
        if len(self.memory) < batch_size:
            return

        cum_loss = 0

        for b in range(num_batches):
            batch = self.memory.sample(batch_size)
            batch = Transition(*zip(*batch))

            nonfinal_next = [batch.next_state[i] for i, done in enumerate(batch.done) if done]
            nonfinal_next = torch.cat(nonfinal_next, dim=1)

            state = torch.stack([*batch.state], dim=1)
            actions = torch.stack([*batch.action], dim=1)
            rewards = torch.stack([*batch.reward], dim=1)

            # get score with policy network
            experienced_values = self.model(state).gather(1, actions)

            # get score with target network
            target_values = torch.zeros(batch_size, device=self.device)
            target_values[batch.done] = self.target_model(nonfinal_next).max(1)[0].detach()
            target_values = (target_values * self.gamma) + rewards

            # compute loss
            loss = self.loss_func(experienced_values, target_values.unsqueeze(1))

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.clampgrad(self.model.parameters())
            self.optimizer.step()

            cum_loss += loss.item()

        return cum_loss




