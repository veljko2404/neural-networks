import warnings
from abc import abstractmethod

from layers.function import Function
from loss_functions.abstract_loss_function import Optimizer


class AdaptiveObject(Function):
    """
    AdaptiveObject instances are functions that can adjust their internal parameters or components. Each subclass must
    implement update_parameters.

    Parameter updates are performed by an optimizer (a class derived from AbstractOptimizer).
    """

    def __init__(self, name: str = 'unnamed', optimizer: Optimizer = None):
        super().__init__(name)
        self._optimizer = optimizer

    @abstractmethod
    def update_parameters(self):
        pass

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        """
        We want a flexible system where different layers may use different optimizers, but we also want the option to assign one
        optimizer to all adaptive layers at once. To avoid manually setting an optimizer for every layer, the Network class will
        later provide a way to broadcast an optimizer to all layers.

        The force parameter allows overriding an already assigned optimizer. This lets us, for example, set a custom optimizer
        for layer 5, then assign another optimizer to the whole network without overwriting layer 5 unless force=True.

        (A simpler version would ignore this logic and just set self._optimizer = deepcopy(optimizer).)
        """
        if force or self._optimizer is None:
            self._optimizer = optimizer
        else:
            warnings.warn('Optimizer already set!')
