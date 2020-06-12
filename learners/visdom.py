import subprocess
import numpy as np
import visdom
import os

EXTENSIONS = ['.mp4']

def launch_visdom(use_visdom=True, port=8067):
    if use_visdom or use_visdom is None:
        try:
            subprocess.Popen('python -m visdom.server -port %d' % port)
        except ImportError:
            pass

class VisdomDictPlotter:

    def __init__(self, env_name='main', session=None):
        if session is None:
            self.viz = visdom.Visdom()
        else:
            self.viz = session
        self.env = env_name
        self.plots = {}

    # dict should be formatted e.g. {'loss' : {'gan_loss': x, 'discriminator': y}}

    # from https://github.com/noagarcia/visdom-tutorial
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
            for k, v in lines_dict.items():
                if not len(v):
                    continue
                self.plots[name] = self.viz.line(X=np.array([0, 0]), Y=np.array([v[0], v[0]]),
                                                   env=self.env, opts=dict(
                            ylabel=k,
                            title=name,
                            xlabel='Epochs',
                        ))
        else:
            for k, v in lines_dict.items():
                if not len(v):
                    continue
                self.viz.line(Y=np.array([v[-1], v[-1]]), X=np.array([len(v), len(v)]),
                              env=self.env, win=self.plots[name], update='append')

class VisdomVideoPlotter:

    # savepath is path to video to render. By default none and will infer
    def __init__(self, env_name, savepath=None, session=None):
        if session is None:
            self.viz = visdom.Visdom()
        else:
            self.viz = session
        self.env = env_name
        self.savepath = savepath
        self.loaded = False

    def update(self, path=None):
        return # TODO: fix!
        if path is None:
            return
        else:
            for file in os.listdir(path):
                if any([file.endswith(ext) for ext in EXTENSIONS]):
                    video = file
        print('loading recent sim from %s' % os.path.join(path, video))
        if self.loaded == False:
            self.video = self.viz.video(videofile=os.path.join(path, video), env=self.env)
            self.loaded = True
        else:
            self.viz.close(self.video)
            self.video = self.viz.video(videofile=os.path.join(path, video), env=self.env)