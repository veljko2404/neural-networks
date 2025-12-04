from abc import abstractmethod

from backend.backend import xp
from layers.function import Function


class ActivationFunction(Function):
    """
    An activation function must be able to transform its inputs and provide its derivative with respect to those inputs.

    For all activation functions except Softmax, the i-th output depends only on the i-th input. In the general case, the
    derivative would be a Nb × n × n tensor (Nb = batch size, n = number of features), representing a batch of Jacobian
    matrices. Under the assumption above, each Jacobian is diagonal (except for Softmax).

    During backpropagation, the gradient w.r.t. the activation’s output is multiplied by this derivative. Since multiplying a
    row by a diagonal matrix reduces to elementwise multiplication with the diagonal, implementations usually exploit this.
    Therefore, derivative returns an array with the same shape as the input, and the actual elementwise product is applied
    later.
    """

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @abstractmethod
    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        pass

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        from layers.activation_functions.softmax import Softmax

        if isinstance(self, Softmax):
            dEdO = dEdO.reshape((dEdO.shape[0], 1, dEdO.shape[-1]))
            dEdI = xp.matmul(dEdO, self.deriv(self._inputs))
            return dEdI.reshape((dEdI.shape[0], dEdI.shape[-1]))

        return xp.multiply(dEdO, self.deriv(self._inputs))

