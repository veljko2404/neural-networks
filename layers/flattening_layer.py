from backend.backend import xp
from layers.function import Function


class FlatteningLayer(Function):
    """
    In a convolutional network, after the convolutional, activation, and optionally pooling layers,
    we apply the "standard" layers we implemented earlier—fully connected layers, activation layers, and loss functions.
    The output of a convolutional (or pooling) layer is a 4D tensor, while the rest of the network
    operates on 2D matrices. Therefore, after all convolutional/pooling layers, we need a layer
    that transforms this 4D output into a 2D matrix. This layer performs exactly that task.
    """
    def __init__(self, name: str = "Flattening Layer"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        return dEdO.reshape(self._inputs.shape)
