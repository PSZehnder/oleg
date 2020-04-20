import gym
from .basesimulator import Simulator
import torch

class GymSimulator(Simulator):

    def __init__(self, name):
        self.name = name
        self.sim = gym.make(name)
        self.reset()

    def update(self, action):
        assert action in self.sim.action_space, 'action: %s not in action space' % action
        state, reward, done, _ = self.sim.step(action)
        self.done = done
        self.state = state
        return reward

    def reset(self):
        self.state = self.sim.reset()
        self.done = False

    def render(self, model, path=None):
        if path:
            self.sim = gym.wrappers.Monitor(self.sim, path)
        if not isinstance(model, torch.nn.Module):
            model = torch.load(model)
        while not self.done:
            action = model(self.state)
            self.update(action)
        self.sim.close()
        self.sim = self.sim.make(self.name)

    def rand_action(self):
        return self.sim.action_space.sample()