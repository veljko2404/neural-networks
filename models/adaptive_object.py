import warnings
from abc import abstractmethod

from layers.function import Function
from optimizers.abstract_optimizer import Optimizer


class AdaptiveObject(Function):

    def __init__(self, name: str = 'unnamed', optimizer: Optimizer = None):
        super().__init__(name)
        self._optimizer = optimizer

    @abstractmethod
    def update_parameters(self):
        pass

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        if force or self._optimizer is None:
            self._optimizer = optimizer
        else:
            warnings.warn('Optimizer already set!')
