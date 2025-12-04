from backend.backend import xp
from abc import ABC, abstractmethod


class Function(ABC):
    """
    This abstract class is one of the key building blocks, and many other classes are based on it (network layers, loss
    functions, etc.). Its job is to turn inputs into outputs and to support backward error propagation.

    It works with n-dimensional tensors. To keep things fast, memory is allocated only once and reused, which makes the code
    a bit more complex. We often use temporary arrays and the `out` parameter to avoid creating extra intermediate data.
    Details like mini-batch size or the exact input/output dimensions of a layer are intentionally hidden from the user.

    We focus on simple, sequential neural networks where each layer has one input and one output. Each function stores its
    input and the gradient with respect to that input, and forward propagation writes results into a provided tensor. In
    older versions this class was called `AbstractLayer`; now the equivalent is `Function`.
    """

    def __init__(self, name: str = 'unnamed'):
        self._training = True
        self._inputs: xp.ndarray = None  # remembering last inout
        self.name = name

    @abstractmethod
    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        pass

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val

    @property
    def parameters(self) -> tuple:
        return tuple()

    @parameters.setter
    def parameters(self, params: tuple):
        pass

    def forward(self, inputs: xp.ndarray) -> xp.ndarray:
        """
        Performs forward propagation on the given inputs.
        and then computes the output.
        """
        self._pre_fw(inputs)

        return self(inputs)

    def _pre_fw(self, inputs: xp.ndarray):
        if self._training:
            self._inputs = inputs

    @abstractmethod
    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        """
        dEdO is an n-dimensional array containing the partial derivatives of the loss with respect to this layer’s outputs.
        This layer cannot compute those derivatives itself — they are produced by the next layer and passed backward.

        The task of this layer is to compute the partial derivatives of the loss with respect to its inputs, which are the
        outputs of the previous layer. This method should also compute any other gradients relevant to this layer, such as the
        gradients of its weights.
        """
        pass
