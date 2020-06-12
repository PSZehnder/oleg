import abc

class Simulator(abc.ABC):

    @abc.abstractmethod
    def update(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def render(self, model, path=None):
        pass

    @abc.abstractmethod
    def rand_action(self):
        pass

    def preprocess(self, state):
        return state

