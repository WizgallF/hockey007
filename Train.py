from abc import ABC, abstractmethod


class Training(ABC):
    def __init__(self, verbose=False):
        self.statistics = None
        self.verbose = verbose

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_self_play(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass
