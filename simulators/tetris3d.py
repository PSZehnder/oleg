import os
import torch
from .basesimulator import Simulator
from .tetris import *


class TetrisSimulator(Simulator):

    def __init__(self, board_size):
        self.board_size = board_size
        self.sim = TetrisInstance(board_size)
        self.reset()

    def update(self, action):
        action = self.sim.action_space[action]
        assert action in self.sim.action_space, 'action: %s not in action space' % action
        state, reward, done = self.sim.update(action)
        self.done = done
        self.state = state
        return reward

    def reset(self):
        self.state = self.sim.reset()
        self.done = False

    def render(self, model, path=None):

        frames_queue = []

        if self.sim.dimension == 2:
            fig = plt.figure()
            ax = plt.axes()
        else:
            fig, ax = render.init3dplot(self.sim.board_extents)

        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            self.reset()
        if not isinstance(model, torch.nn.Module):
            model = torch.load(model)

        while not self.done:
            action = torch.argmax(model(self.preprocess(self.state)), dim=0).cpu().numpy()
            self.update(action)
            frames_queue.append(self.sim.board_with_piece)

        if self.sim.dimension == 2:
            render2d(frames_queue, fig, ax, path)
        else:
            render3d(frames_queue, fig, ax, path)

        self.reset()

    def rand_action(self):
        return sample(self.sim.action_space)
