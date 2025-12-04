from abc import abstractmethod

from backend.backend import xp
from layers.function import Function


class LossFunction(Function):

    def __init__(self, name: str = 'loss'):
        super().__init__(name)

    def forward(self, inputs: xp.ndarray) -> xp.ndarray:
        raise Exception("Not implemented!")

    @abstractmethod
    def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
        """Call will return value of error function"""
        pass

    @abstractmethod
    def backward(self, y: xp.ndarray, t: xp.ndarray) -> xp.ndarray:
        """
        Backward propagation starts from this “layer.” Python allows method overriding with different argument names, so unlike
        other layers, the input tensor here is called target instead of dEdO, to emphasize that it represents the desired output
        values.

        The result of this call is the set of partial derivatives dE/dy.
        """
        pass
