import subprocess
import numpy as np
import visdom

def launch_visdom(use_visdom=True, port=8067):
    if use_visdom or use_visdom is None:
        try:
            subprocess.Popen('python -m visdom.server -port %d' % port)
        except ImportError:
            pass

# from https://github.com/noagarcia/visdom-tutorial
class VisdomDictPlotter:

    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    # dict should be formatted e.g. {'loss' : {'gan_loss': x, 'discriminator': y}}
    def plot(self, dct):
        for k, v in dct.items():
            if isinstance(v, dict):
                self._plot_multiple(k, v)
            else:
                if k not in self.plots:
                    self.plots[k] = self.viz.lines(X=np.array[0, 0], Y=np.array([v[0], v[0]]),
                                                   env=self.env, opts=dict(
                            title=k,
                            xlabel='Epochs',
                            ylabel=k
                        ))
                else:
                    self.viz.line(X=np.array([len(v)]), Y=np.array([v[-1]]), env=self.env,
                                  win=self.plots[k], update='append')

    def _plot_multiple(self, name, lines_dict):
        if name not in self.plots:
            for k, v in lines_dict:
                self.plots[name] = self.viz.lines(X=np.array[0, 0], Y=np.array([v[0], v[0]]),
                                                   env=self.env, opts=dict(
                            legend=[k],
                            title=name,
                            xlabel='Epochs',
                        ))
        else:
            for k, v in lines_dict:
                self.viz.line(Y=np.array([v[-1], v[-1]]), X=np.array([len(v), len(v)]),
                              env=self.env, win=self.plots[name], update='append')