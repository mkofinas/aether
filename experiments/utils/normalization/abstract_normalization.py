import os
from abc import ABCMeta, abstractmethod


class AbstractNormalization(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def normalize(self, x):
        pass

    def normalize_positions(self, x):
        return self.normalize(x)

    def normalize_velocities(self, x):
        return self.normalize(x)

    @abstractmethod
    def unnormalize(self, x):
        pass

    def unnormalize_positions(self, x):
        return self.unnormalize(x)

    def unnormalize_velocities(self, x):
        return self.unnormalize(x)

    def torch_unnormalize(self, x):
        return self.unnormalize(x)

