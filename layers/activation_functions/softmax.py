from backend.backend import np
from layers.activation_functions.activation_function import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self, name: str = "Softmax"):
        super().__init__(name)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        y = np.exp(inputs)
        tmp = np.sum(y, axis=-1, keepdims=True)
        return np.divide(y, tmp)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        y = self(x)
        dx = np.zeros((y.shape[0], y.shape[1], y.shape[1]))
        for batch_index in range(y.shape[0]):
            dx[batch_index, :, :] = -np.matmul(y[batch_index, :].reshape(-1, 1),
                                               y[batch_index, :].reshape(-1, 1).T)
            dx[batch_index, :, :] += np.diagflat(y[batch_index, :])

        return dx


