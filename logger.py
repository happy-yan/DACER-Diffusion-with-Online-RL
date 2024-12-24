import datetime
import numpy as np
import pickle
import os

class Logger(object):

    def __init__(self, name, keys, max_epochs=10000, times=3, config=None, path=None, id=None):
        self.tracker = {key: np.zeros((max_epochs*times,)) for key in keys}
        self.name = name
        self.pointer = 0
        self.max_epochs = max_epochs
        self.times = times
        if config is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/config_{id}.pkl', "wb") as f:
                pickle.dump(config, f)

    def add(self, **kwargs):
        if self.pointer >= self.max_epochs * self.times:
            raise StopIteration("logger is full!")

        for key in kwargs.keys():
            self.tracker[key][self.pointer] = kwargs[key]

        self.pointer += 1

    def save(self, path, id):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, id), "wb") as f:
            pickle.dump(self, f)