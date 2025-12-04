from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """
    f(x_i) =  exp(x_i) / sum_j exp(x_j), j from {1, 2, 3, ... , n}

    With the Softmax function, each input affects all n outputs, so the derivative is more complex:
    dFj/dXi = Si(1 − Sj) for i = j, and −SiSj otherwise, where Si = Softmax(xi).

    Softmax is typically used in the final layer of a multiclass classifier. Its outputs are probabilities over the n
    classes, and the usual loss function for this setting is cross-entropy.

    Softmax has no trainable parameters, so its full Jacobian is only needed for backpropagation. For the common case where
    the last layer is Softmax and the loss is cross-entropy, the gradient simplifies to dE/dX = output − target.

    Because of this simplification, a practical implementation often skips explicit backward computation for Softmax and
    directly returns output − target.
    """

    def __init__(self, name: str = "Softmax"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        y = xp.exp(inputs)
        tmp = xp.sum(y, axis=-1, keepdims=True)
        return xp.divide(y, tmp)

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        y = self(x)
        dx = xp.zeros((y.shape[0], y.shape[1], y.shape[1]))
        for batch_index in range(y.shape[0]):
            dx[batch_index, :, :] = -xp.matmul(y[batch_index, :].reshape(-1, 1),
                                               y[batch_index, :].reshape(-1, 1).T)

            dx[batch_index, :, :] += xp.diagflat(y[batch_index, :])

        return dx


