from backend.backend import xp

from layers.activation_functions.activation_function import ActivationFunction


class ReLU(ActivationFunction):
    """
    ReLU function is defined as f(x) = max(x, 0)
    So the function behaves like the identity for x > 0 and like the constant 0 for x ≤ 0.
    The derivative of the identity is 1, and the derivative of a constant is 0.
    """

    def __init__(self, name="ReLU"):
        super().__init__(name=name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return xp.maximum(inputs, 0)

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        df = xp.zeros_like(x)
        df[x > 0] = 1
        return df
