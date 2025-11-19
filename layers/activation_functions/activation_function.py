from abc import abstractmethod

from backend.backend import np
from layers.function import Function


class ActivationFunction(Function):
    def __init__(self, name: str = None):
        super().__init__(name=name)

    @abstractmethod
    def deriv(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, dEdO: np.ndarray) -> np.ndarray:
        from layers.activation_functions.softmax import Softmax

        if isinstance(self, Softmax):
            dEdO = dEdO.reshape((dEdO.shape[0], 1, dEdO.shape[-1]))
            dEdI = np.matmul(dEdO, self.deriv(self._inputs))
            return dEdI.reshape((dEdI.shape[0], dEdI.shape[-1]))

        return np.multiply(dEdO, self.deriv(self._inputs))

