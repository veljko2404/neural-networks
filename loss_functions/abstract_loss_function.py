from abc import ABC, abstractmethod

from backend.backend import xp


class Optimizer(ABC):
    """
    The optimizer’s goal is to update the parameters of an AdaptiveObject (see that class). Every concrete optimizer must
    implement update_parameters, which receives parameters and their gradients and updates them accordingly.

    Regarding neural-network optimization:

    Training a neural network has several specific challenges. Gradients are usually estimated from small mini-batches, so
    choosing α via line-search (common in classical optimization) is ineffective — an alpha that works for a few samples may not
    generalize. alpha must avoid being too small (slow learning) or too large (unstable updates).

    The loss function often contains flat regions (plateaus) where gradients are near zero; the optimizer must still be able
    to escape these areas.

    During training, some synapses specialize in rare features—certain neurons activate only for specific examples. It is
    useful if the optimizer can learn faster from such rare but informative signals.
    """

    def __init__(self, lr: float):
        self.lr = lr

    @abstractmethod
    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        pass
