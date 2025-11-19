from backend.backend import np
from layers.activation_functions.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    def __init__(self, name: str = "Sigmoid"):
        super().__init__(name)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))

    def deriv(self, x: np.ndarray) -> np.ndarray:
        y = self(x)
        return y * (1 - y)
